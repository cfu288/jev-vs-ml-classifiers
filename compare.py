import asyncio
import hashlib
import json
import os
import platform
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from scipy.special import softmax
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_wine
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed,
)
from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy

# === Settings ===

DATASETS = ["diabetes", "titanic", "wine", "diabetes130", "adult"]
MODELS = ["api", "sklearn", "bart", "modernbert"]
# Prompt wordings for the zero-shot models. zero_shot_task describes each one.
VARIANTS = ["list", "short", "structured"]
SEED = 42
KFOLD_BELOW_ROWS = 2000
CONCURRENCY = 16
RESULTS_DIR = Path(__file__).parent / "results"

# === Data Loaders ===


@dataclass
class Dataset:
    """A labeled dataset and the wording used to turn its rows into text.

    Attributes:
        name (str): Short id used in file names and results.csv.
        X (pd.DataFrame): Features with one row per example.
        y (pd.Series): Label of each row.
        readable (dict[str, str]): Plain English name for each column of X. The row text lists columns in this dict's order.
        subject (str): First line of every row text. It says what one row describes.
        questions (dict[str, str]): Question asked about each row for each prompt variant.
        options (dict[str, str]): Description of each label. Zero-shot models answer with one of the
            labels, and the descriptions tell them what each label means.
        group (pd.Series | None): Id that keeps related rows on the same side of a split.
            Readmission uses the patient id. None for the other datasets.
    """

    name: str
    X: pd.DataFrame
    y: pd.Series
    readable: dict[str, str]
    subject: str
    questions: dict[str, str]
    options: dict[str, str]
    group: pd.Series | None = None


def load_diabetes() -> Dataset:
    """Load the Pima Indians Diabetes data of 768 women. The label is diabetes onset within five years.

    Missing glucose, blood pressure, skin fold, insulin and BMI values are recorded as 0 in the source data.
    We normalize those to NaN.

    Returns:
        Dataset: The diabetes data with its prompt text.
    """
    X, y = fetch_openml(
        "diabetes", version=1, as_frame=True, parser="auto", return_X_y=True
    )
    X = X.copy()
    zero_means_missing = ["plas", "pres", "skin", "insu", "mass"]
    X[zero_means_missing] = X[zero_means_missing].replace(0, np.nan)
    y = y.map({"tested_negative": "not_diabetic", "tested_positive": "diabetic"})
    readable = {
        "preg": "Number of pregnancies",
        "plas": "Plasma glucose concentration (mg/dL, 2-hour oral glucose tolerance test)",
        "pres": "Diastolic blood pressure (mm Hg)",
        "skin": "Triceps skin fold thickness (mm)",
        "insu": "2-hour serum insulin (mu U/ml)",
        "mass": "Body mass index",
        "pedi": "Diabetes pedigree function",
        "age": "Age (years)",
    }
    return Dataset(
        name="diabetes",
        X=X,
        y=y,
        readable=readable,
        subject="Diagnostic measurements for an adult female patient, aged 21 or older.",
        questions={
            "list": "Based on these diagnostic measurements, will this patient develop diabetes within the next five years?",
            "short": "Diabetes within five years?",
            "structured": "You are reviewing a patient's diagnostic record. Classify whether the patient will develop diabetes within five years of the exam.",
        },
        options={
            "diabetic": "The patient will develop diabetes within five years",
            "not_diabetic": "The patient will not develop diabetes within five years",
        },
    )


def load_titanic() -> Dataset:
    """Load the Titanic survival data of 1,309 passengers. The label is survived or died.

    Only the seven features known before the sinking are kept. These are ticket class, sex, age,
    siblings or spouses aboard, parents or children aboard, fare and port of embarkation.

    Returns:
        Dataset: The Titanic data with its prompt text.
    """
    X, y = fetch_openml(
        "titanic", version=1, as_frame=True, parser="auto", return_X_y=True
    )
    cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    X = X[cols].copy()
    y = y.map({"0": "died", "1": "survived"})
    readable = {
        "pclass": "Ticket class (1 = first, 2 = second, 3 = third)",
        "sex": "Sex",
        "age": "Age (years)",
        "sibsp": "Number of siblings or spouses aboard",
        "parch": "Number of parents or children aboard",
        "fare": "Ticket fare (pounds)",
        "embarked": "Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)",
    }
    return Dataset(
        name="titanic",
        X=X,
        y=y,
        readable=readable,
        subject="A passenger aboard a passenger ship that sank.",
        questions={
            "list": "Based only on the listed attributes, predict whether this passenger survived the sinking.",
            "short": "Did this passenger survive?",
            "structured": "You are reviewing a passenger record from a ship that sank. Classify whether this passenger survived.",
        },
        options={
            "survived": "The passenger survived",
            "died": "The passenger did not survive",
        },
    )


def load_wine_cultivars() -> Dataset:
    """Load sklearn's Wine dataset of 178 wines from three grape varieties.

    The 13 features are chemical measurements. The label is Barolo, Grignolino or Barbera.

    Returns:
        Dataset: The wine data with its prompt text.
    """
    X, y = load_wine(as_frame=True, return_X_y=True)
    # sklearn labels the wines 0, 1, 2. Their row counts (59, 71, 48) match Barolo, Grignolino and Barbera in Forina 1986: https://www.rdocumentation.org/packages/whitening/versions/1.4.0/topics/forina1986
    y = y.map({0: "barolo", 1: "grignolino", 2: "barbera"})
    readable = {c: c.replace("_", " ").capitalize() for c in X.columns}
    readable["od280/od315_of_diluted_wines"] = "OD280/OD315 ratio of diluted wines"
    return Dataset(
        name="wine",
        X=X,
        y=y,
        readable=readable,
        subject="Chemical analysis of a wine.",
        questions={
            "list": "Based on this chemical analysis, which of these three wines is it?",
            "short": "Which wine is this?",
            "structured": "You are reviewing a chemical analysis of a wine. Classify which of the three wines it is.",
        },
        options={
            "barolo": "Barolo wine",
            "grignolino": "Grignolino wine",
            "barbera": "Barbera wine",
        },
    )


# Text for the numeric admission type, discharge disposition and admission source codes from the dataset's IDS_mapping.csv (UCI archive id 296).
UCI_CODE_MAPS = {
    "admission_type_id": {
        1: "Emergency",
        2: "Urgent",
        3: "Elective",
        4: "Newborn",
        5: "Not Available",
        6: "NULL",
        7: "Trauma Center",
        8: "Not Mapped",
    },
    "discharge_disposition_id": {
        1: "Discharged to home",
        2: "Discharged/transferred to another short term hospital",
        3: "Discharged/transferred to SNF",
        4: "Discharged/transferred to ICF",
        5: "Discharged/transferred to another type of inpatient care institution",
        6: "Discharged/transferred to home with home health service",
        7: "Left AMA",
        8: "Discharged/transferred to home under care of Home IV provider",
        9: "Admitted as an inpatient to this hospital",
        10: "Neonate discharged to another hospital for neonatal aftercare",
        11: "Expired",
        12: "Still patient or expected to return for outpatient services",
        13: "Hospice / home",
        14: "Hospice / medical facility",
        15: "Discharged/transferred within this institution to Medicare approved swing bed",
        16: "Discharged/transferred/referred another institution for outpatient services",
        17: "Discharged/transferred/referred to this institution for outpatient services",
        18: "NULL",
        19: "Expired at home. Medicaid only, hospice.",
        20: "Expired in a medical facility. Medicaid only, hospice.",
        21: "Expired, place unknown. Medicaid only, hospice.",
        22: "Discharged/transferred to another rehab fac including rehab units of a hospital",
        23: "Discharged/transferred to a long term care hospital",
        24: "Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare",
        25: "Not Mapped",
        26: "Unknown/Invalid",
        27: "Discharged/transferred to a federal health care facility",
        28: "Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital",
        29: "Discharged/transferred to a Critical Access Hospital (CAH)",
        30: "Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere",
    },
    "admission_source_id": {
        1: "Physician Referral",
        2: "Clinic Referral",
        3: "HMO Referral",
        4: "Transfer from a hospital",
        5: "Transfer from a Skilled Nursing Facility (SNF)",
        6: "Transfer from another health care facility",
        7: "Emergency Room",
        8: "Court/Law Enforcement",
        9: "Not Available",
        10: "Transfer from critical access hospital",
        11: "Normal Delivery",
        12: "Premature Delivery",
        13: "Sick Baby",
        14: "Extramural Birth",
        15: "Not Available",
        17: "NULL",
        18: "Transfer From Another Home Health Agency",
        19: "Readmission to Same Home Health Agency",
        20: "Not Mapped",
        21: "Unknown/Invalid",
        22: "Transfer from hospital inpt/same fac reslt in a sep claim",
        23: "Born inside this hospital",
        24: "Born outside this hospital",
        25: "Transfer from Ambulatory Surgery Center",
        26: "Transfer from Hospice",
    },
}


def load_diabetes130() -> Dataset:
    """Load the Diabetes 130-US hospitals data of about 99,000 hospital stays. The label is readmission within 30 days.

    Stays that ended in death or hospice are removed because those patients cannot be readmitted.
    Numeric admission type, discharge disposition and admission source codes are replaced with their text.
    The patient id is kept as the group, so one patient's stays never appear in both train and test.

    Returns:
        Dataset: The readmission data with its prompt text.
    """
    X_all, y_all = fetch_openml(
        "Diabetes130US", version=1, as_frame=True, parser="auto", return_X_y=True
    )
    alive = ~X_all["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])
    frame = X_all[alive]
    group = frame["patient_nbr"]
    # examide and citoglipton are "No" in every row, so they are dropped.
    X = frame.drop(columns=["encounter_id", "patient_nbr", "examide", "citoglipton"])
    cat_cols = X.select_dtypes(exclude="number").columns
    X[cat_cols] = X[cat_cols].astype(object).replace("?", np.nan)
    code_cols = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
    for col in code_cols:
        X[col] = X[col].map(UCI_CODE_MAPS[col]).fillna("Unknown/Invalid")
    X["max_glu_serum"] = X["max_glu_serum"].replace("None", "Not tested")
    # Table 1 of Strack et al. 2014 defines the ranges the raw A1C result codes stand for (https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/)
    X["A1Cresult"] = X["A1Cresult"].replace(
        {"None": "Not tested", "Norm": "Normal (<7%)", ">7": "7-8%", ">8": ">8%"}
    )
    y = (y_all[alive] == "<30").map({True: "readmitted", False: "not_readmitted"})
    med_cols = [
        c
        for c in X.columns
        if X[c].notna().any()
        and X[c].dropna().isin(["No", "Steady", "Up", "Down"]).all()
        and c not in code_cols
    ]
    readable = {
        "race": "Race",
        "gender": "Gender",
        "age": "Age bracket (years)",
        "weight": "Weight bracket (lbs)",
        "admission_type_id": "Admission type",
        "discharge_disposition_id": "Discharge disposition",
        "admission_source_id": "Admission source",
        "time_in_hospital": "Days in hospital",
        "payer_code": "Insurance payer code",
        "medical_specialty": "Admitting physician specialty",
        "num_lab_procedures": "Lab tests during encounter",
        "num_procedures": "Procedures during encounter",
        "num_medications": "Distinct medications administered",
        "number_outpatient": "Outpatient visits in prior year",
        "number_emergency": "Emergency visits in prior year",
        "number_inpatient": "Inpatient visits in prior year",
        "diag_1": "Primary diagnosis (ICD-9 code)",
        "diag_2": "Secondary diagnosis (ICD-9 code)",
        "diag_3": "Additional diagnosis (ICD-9 code)",
        "number_diagnoses": "Number of diagnoses recorded",
        "max_glu_serum": "Max glucose serum test result",
        "A1Cresult": "HbA1c test result",
        "change": "Any diabetes medication changed (Ch = changed)",
        "diabetesMed": "Any diabetes medication prescribed",
    }
    for c in med_cols:
        readable[c] = f"{c.replace('.', ' + ').capitalize()} (No / Steady / Up / Down)"
    return Dataset(
        name="diabetes130",
        group=group,
        X=X,
        y=y,
        readable={c: readable[c] for c in X.columns},
        subject=(
            "A US hospital inpatient encounter of 1-14 days for a patient with "
            "diabetes. For each medication, Up/Down means the dose was increased/"
            "decreased during the encounter, Steady means unchanged, No means not "
            "prescribed."
        ),
        questions={
            "list": "Based on this discharge record, predict whether an inpatient readmission will be recorded within 30 days of discharge.",
            "short": "Readmitted within 30 days?",
            "structured": "You are reviewing a hospital discharge record. Classify whether an inpatient readmission will be recorded within 30 days of discharge.",
        },
        options={
            "readmitted": "A readmission will be recorded within 30 days of discharge",
            "not_readmitted": "No readmission will be recorded within 30 days of discharge",
        },
    )


def load_adult() -> Dataset:
    """Load the UCI Adult census data of 48,842 people. The label is annual income above $50,000.

    Returns:
        Dataset: The census data with its prompt text.
    """
    X, y = fetch_openml(
        "adult", version=2, as_frame=True, parser="auto", return_X_y=True
    )
    # fnlwgt is a survey weight and not a trait of the person. education-num repeats education. Both are dropped.
    X = X.drop(columns=["fnlwgt", "education-num"])
    y = y.map({"<=50K": "at_most_50k", ">50K": "over_50k"})
    readable = {
        "age": "Age (years)",
        "workclass": "Employer type",
        "education": "Highest education level",
        "marital-status": "Marital status",
        "occupation": "Occupation",
        "relationship": "Household relationship",
        "race": "Race",
        "sex": "Sex",
        "capital-gain": "Capital gains (USD)",
        "capital-loss": "Capital losses (USD)",
        "hours-per-week": "Work hours per week",
        "native-country": "Native country",
    }
    return Dataset(
        name="adult",
        X=X,
        y=y,
        readable=readable,
        subject="A person's record from a US census survey.",
        questions={
            "list": "Is this person's reported annual income more than $50,000?",
            "short": "Income over $50,000?",
            "structured": "You are reviewing a census survey record. Classify whether this person's reported annual income is more than $50,000.",
        },
        options={
            "over_50k": "Annual income above $50,000",
            "at_most_50k": "Annual income of $50,000 or less",
        },
    )


# === Sklearn Models ===


class Predictions(NamedTuple):
    """One model's predictions for a set of test rows.

    Attributes:
        labels (pd.Series): Predicted label for each row.
        proba (pd.DataFrame | None): Probability of each class for each row. None when the model gives no probabilities.
        scores (pd.DataFrame | None): SVM score of each class for each row. A higher score means more likely.
            The SVMs give these instead of probabilities. None for every other model.
        train_seconds (float): Seconds spent training. 0 for zero-shot models.
        row_seconds (float): Median seconds to predict one row.
    """

    labels: pd.Series
    proba: pd.DataFrame | None
    scores: pd.DataFrame | None
    train_seconds: float
    row_seconds: float


def holdout(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hold out a random 10% of the data using SEED.

    The proportion of each label can differ slightly between the two parts, because rows are picked
    without looking at their labels.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Labels.
        groups (pd.Series | None): Id for each row. When given, 10% of the ids are held out,
            and all rows of an id land on the same side. None holds out 10% of the rows.

    Returns:
        tuple: X_kept, X_held, y_kept, y_held.
    """
    if groups is None:
        return train_test_split(X, y, test_size=0.1, random_state=SEED)
    kept, held = next(
        GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED).split(
            X, y, groups=groups
        )
    )
    return X.iloc[kept], X.iloc[held], y.iloc[kept], y.iloc[held]


def split_dataset(
    ds: Dataset,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a dataset into 90% train and 10% test.

    Args:
        ds (Dataset): Dataset to split. When ds.group is set, all rows with the same group id stay on one side.

    Returns:
        tuple: X_train, X_test, y_train, y_test.
    """
    return holdout(ds.X, ds.y, ds.group)


def make_folds(ds: Dataset) -> list[tuple[pd.Index, pd.Index]]:
    """Choose the train and test rows of a dataset.

    Datasets under KFOLD_BELOW_ROWS rows use 10-fold cross-validation. The rows are split into 10 parts with
    the same label proportions, and each part is tested once by a model trained on the other 9.
    Larger datasets use one 90/10 split.
    See https://scikit-learn.org/stable/modules/cross_validation.html

    Args:
        ds (Dataset): Dataset to split.

    Returns:
        list[tuple[pd.Index, pd.Index]]: One (train rows, test rows) pair per fold.
    """
    if len(ds.X) < KFOLD_BELOW_ROWS:
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        return [(ds.X.index[tr], ds.X.index[te]) for tr, te in cv.split(ds.X, ds.y)]
    X_train, X_te, _, _ = split_dataset(ds)
    return [(X_train.index, X_te.index)]


def make_classifiers(X: pd.DataFrame) -> dict[str, Pipeline]:
    """Build each sklearn model as a pipeline that preprocesses the data before the model sees it.

    Based on https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

    Args:
        X (pd.DataFrame): Features. Used only to find the number and text columns.

    Returns:
        dict[str, Pipeline]: Untrained pipelines keyed by display name.
    """

    def pipe(clf: BaseEstimator) -> Pipeline:
        """Put the shared preprocessing in front of one classifier."""
        return Pipeline([("prep", make_preprocessor(X)), ("clf", clf)])

    has_categories = X.select_dtypes(include="number").shape[1] < X.shape[1]

    return {
        "Majority class": pipe(DummyClassifier(strategy="most_frequent")),
        "Nearest Neighbors": pipe(KNeighborsClassifier()),
        "Decision Tree": pipe(DecisionTreeClassifier(random_state=SEED)),
        "Random Forest": pipe(RandomForestClassifier(random_state=SEED)),
        "Neural Net": pipe(MLPClassifier(max_iter=1000, random_state=SEED)),
        "AdaBoost": pipe(AdaBoostClassifier(random_state=SEED)),
        "Naive Bayes": pipe(GaussianNB()),
        # QDA's default reg_param=0.0 fails to fit one-hot columns, so datasets with text columns use 0.1.
        "QDA": pipe(
            QuadraticDiscriminantAnalysis(reg_param=0.1 if has_categories else 0.0)
        ),
        "Linear SVM": pipe(SVC(kernel="linear", random_state=SEED)),
        "RBF SVM": pipe(SVC(random_state=SEED)),
    }


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build the preprocessing step that every sklearn model uses.

    Number columns get missing values filled with the column median and are then scaled.
    Text columns get missing values filled with "missing" and are then one-hot encoded.
    Based on https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

    Args:
        X (pd.DataFrame): Features. Used only to find the number and text columns.

    Returns:
        ColumnTransformer: Untrained preprocessing step.
    """
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    numeric_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def time_single_row(
    predict_row: Callable[[pd.DataFrame], object], X: pd.DataFrame
) -> float:
    """Time how long one prediction takes.

    Calls predict_row once without timing to warm up. Then times one call for each of the first 100 rows of X.

    Args:
        predict_row (Callable[[pd.DataFrame], object]): Predicts one row passed in as a one-row DataFrame.
        X (pd.DataFrame): Rows to predict.

    Returns:
        float: Median seconds per prediction.
    """
    predict_row(X.iloc[:1])
    times = []
    for i in range(min(100, len(X))):
        row = X.iloc[[i]]
        start = time.perf_counter()
        predict_row(row)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


# === Zero-Shot APIs ===


@dataclass(frozen=True)
class ApiModel:
    """One API model and how to call it.

    Attributes:
        name (str): Display name in results and charts.
        key (str): Short id used in cache file names.
        provider (str): "jev" or "openai". Picks the request format and the client.
        model (str): Model id sent to the API.
        effort (str | None): OpenAI reasoning effort. None for JEV.
        requests_per_minute (int): Rate limit. Requests are spaced to stay under 90% of it.
    """

    name: str
    key: str
    provider: str
    model: str
    effort: str | None
    requests_per_minute: int


# Rate limits in requests per minute. JEV's is from https://docs.typesafe.ai/models. The OpenAI values are Tier 4 account limits.
API_MODELS = [
    ApiModel(
        name="JEV (zero-shot)",
        key="jev",
        provider="jev",
        model="jev-1.13.0",
        effort=None,
        requests_per_minute=1200,
    ),
    ApiModel(
        name="GPT-5-nano (zero-shot)",
        key="nano",
        provider="openai",
        model="gpt-5-nano-2025-08-07",
        effort="minimal",
        requests_per_minute=10000,
    ),
    ApiModel(
        name="GPT-5-mini (zero-shot)",
        key="mini",
        provider="openai",
        model="gpt-5-mini-2025-08-07",
        effort="minimal",
        requests_per_minute=10000,
    ),
    ApiModel(
        name="GPT-6-luna (zero-shot)",
        key="luna",
        provider="openai",
        model="gpt-6-luna",
        effort="none",
        requests_per_minute=10000,
    ),
]
# Used only by the commented-out JSON-state version of JEV's request in build_request.
# JEV_STATE_HINT = "The record is described in `subject` and its attributes are in `features`."
ATTEMPTS = 3
CACHE_DIR = RESULTS_DIR / "cache"
CHUNK = 500


def format_value(v: object) -> str:
    """Format one value for the row text.

    NaN becomes "unknown". A whole float like 33.0 becomes "33". Other floats keep 4 significant digits.
    Integers and text become their plain text.

    Args:
        v (object): One cell value.

    Returns:
        str: The value as text.
    """
    if pd.isna(v):
        return "unknown"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def row_text(ds: Dataset, row: pd.Series, variant: str) -> str:
    """Write one row as text in the list or short layout.

    The first line is ds.subject. Each feature then follows as "Name: value" in the order of ds.readable.
    The "short" variant puts all features on one line with commas between them.
    The "list" variant puts each feature on its own line starting with "- ".

    Args:
        ds (Dataset): Dataset the row belongs to.
        row (pd.Series): One row of ds.X.
        variant (str): "list" or "short".

    Returns:
        str: The row as text.
    """
    features = [
        f"{label}: {format_value(row[col])}" for col, label in ds.readable.items()
    ]
    if variant == "short":
        return f"{ds.subject}\n" + ", ".join(features)
    return f"{ds.subject}\n" + "\n".join(f"- {feature}" for feature in features)


def json_value(v: object) -> float | int | str | None:
    """Format one value for the JSON record.

    Floats keep 4 significant digits as in format_value, and whole numbers become integers.
    NaN becomes None, which is null in JSON. Text is passed through unchanged.

    Args:
        v (object): One cell value.

    Returns:
        float | int | str | None: The value for the JSON record.
    """
    if pd.isna(v):
        return None
    if isinstance(v, (float, np.floating)):
        rounded = float(v) if float(v).is_integer() else float(f"{v:.4g}")
        return int(rounded) if rounded.is_integer() else rounded
    if isinstance(v, (int, np.integer)):
        return int(v)
    return str(v)


def row_record(ds: Dataset, row: pd.Series) -> dict:
    """Write one row as a JSON object.

    Args:
        ds (Dataset): Dataset the row belongs to.
        row (pd.Series): One row of ds.X.

    Returns:
        dict: The subject under "subject" and each feature's name and value under "features".
    """
    return {
        "subject": ds.subject,
        "features": {label: json_value(row[col]) for col, label in ds.readable.items()},
    }


@dataclass(frozen=True)
class ZeroShotTask:
    """Everything a zero-shot model is given for one row. Used for JEV, OpenAI and BART inputs.

    Attributes:
        question (str): Question asked about the row in the chosen prompt variant.
        options (dict[str, str]): Description of each label.
        record (dict): The row as a JSON object. The structured variant's text is this object written as JSON.
        text (str): The row as text in the chosen variant. Every model receives this.
    """

    question: str
    options: dict[str, str]
    record: dict
    text: str


def zero_shot_task(ds: Dataset, row: pd.Series, variant: str) -> ZeroShotTask:
    """Build the zero-shot input for one row in one prompt variant.

    The three variants in VARIANTS keep the same facts and options but word the prompt differently.
    "list" asks the full question and gives one feature per line.
    "short" asks a short question and gives all features on one line.
    "structured" frames the task as a record review and gives the row as a JSON object written as text.

    Args:
        ds (Dataset): Dataset the row belongs to.
        row (pd.Series): One row of ds.X.
        variant (str): "list", "short" or "structured".

    Returns:
        ZeroShotTask: Question, options and row for that variant.
    """
    record = row_record(ds, row)
    text = json.dumps(record) if variant == "structured" else row_text(ds, row, variant)
    return ZeroShotTask(
        question=ds.questions[variant], options=ds.options, record=record, text=text
    )


def openai_instructions(task: ZeroShotTask) -> str:
    """Write the instructions field of an OpenAI request.

    The field holds the same question and options that JEV gets in its own fields.

    Args:
        task (ZeroShotTask): Input for one row.

    Returns:
        str: The question, a line asking for one label, then one "- label: description" line per option.
    """
    options = "\n".join(f"- {label}: {desc}" for label, desc in task.options.items())
    return f"{task.question}\nAnswer with exactly one of the labels below.\nOptions:\n{options}"


def build_request(api: ApiModel, task: ZeroShotTask) -> dict:
    """Build the API request for one row with each API's own input fields.

    JEV gets the question, options and row text in separate fields.
    OpenAI gets the question and options as instructions and the row text as input.
    The request holds the model id, the model settings, the question, the options and the row text.
    The request's hash is the cache key, so changing any of these means new API calls.

    Args:
        api (ApiModel): Model to call.
        task (ZeroShotTask): Input for one row.

    Returns:
        dict: The request. call_jev and call_openai send it.
    """
    if api.provider == "jev":
        return {
            "provider": "jev",
            "model": api.model,
            "instructions": task.question,
            "criteria": task.options,
            "state": task.text,
            # JEV's docs recommend a JSON object state that the question names. JEV scored lower with it on the numeric datasets here, so it is off.
            # "instructions": f"{task.question} {JEV_STATE_HINT}",
            # "state": task.record,
        }
    return {
        "provider": "openai",
        "model": api.model,
        "effort": api.effort,
        "instructions": openai_instructions(task),
        "input": task.text,
        "schema": {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": list(task.options)}},
            "required": ["label"],
            "additionalProperties": False,
        },
    }


def request_hash(request: dict) -> str:
    """SHA-256 hash of the request written as JSON. collect uses it as the cache key."""
    return hashlib.sha256(json.dumps(request).encode()).hexdigest()


def make_client(api: ApiModel) -> AsyncTypeSafeClient | AsyncOpenAI:
    """Create the client for one API with its own retries turned off, so send() can time only the successful try.

    Args:
        api (ApiModel): Model to call.

    Returns:
        AsyncTypeSafeClient | AsyncOpenAI: Client to pass to send().
    """
    if api.provider == "jev":
        return AsyncTypeSafeClient(
            api_key=os.environ["TYPESAFE_AI_KEY"], retry=RetryPolicy(max_retries=0)
        )
    # The SDK's 600 s default timeout lets one stuck request stall a whole batch, so requests time out after 30 s.
    return AsyncOpenAI(api_key=os.environ["OPEN_AI_KEY"], max_retries=0, timeout=30.0)


async def call_jev(client: AsyncTypeSafeClient, request: dict) -> dict:
    """Send one request to JEV.

    Args:
        client (AsyncTypeSafeClient): JEV client.
        request (dict): Request from build_request.

    Returns:
        dict: Keys label (the chosen option), probabilities (one per option), confidence (JEV's 0 to 1 score of how sure
            it is), model (the version the API reports), input_tokens and output_tokens.
    """
    question = Choice(
        instructions=request["instructions"], criteria=request["criteria"]
    )
    resp = await client.system_one(
        state=request["state"], questions={"label": question}, model=request["model"]
    )
    answer = resp.choices["label"]
    return {
        "label": answer.choice,
        "probabilities": answer.probabilities,
        "confidence": answer.confidence,
        "model": resp.model,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }


async def call_openai(client: AsyncOpenAI, request: dict) -> dict:
    """Send one request to OpenAI.

    Args:
        client (AsyncOpenAI): OpenAI client.
        request (dict): Request from build_request.

    Returns:
        dict: Keys label (the chosen option), probabilities (always None), model (the version the API reports),
            input_tokens and output_tokens.

    Raises:
        RuntimeError: The response was incomplete or refused.
    """
    resp = await client.responses.create(
        model=request["model"],
        store=False,
        instructions=request["instructions"],
        input=request["input"],
        reasoning={"effort": request["effort"]},
        text={
            "format": {
                "type": "json_schema",
                "name": "classification",
                "schema": request["schema"],
                "strict": True,
            }
        },
    )
    if resp.status != "completed" or not resp.output_text or resp.usage is None:
        raise RuntimeError(
            f"OpenAI returned no answer (status {resp.status}), possibly a refusal"
        )
    return {
        "label": json.loads(resp.output_text)["label"],
        "probabilities": None,
        "model": resp.model,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }


async def send(client: AsyncTypeSafeClient | AsyncOpenAI, request: dict) -> dict:
    """Send one request and try up to ATTEMPTS times in total.

    Any error triggers a retry. The wait before try n+1 is 2 to the power n seconds, so 2 s then 4 s.
    Latency covers only the successful try, so the waiting is not timed.

    Args:
        client (AsyncTypeSafeClient | AsyncOpenAI): Client from make_client.
        request (dict): Request from build_request.

    Returns:
        dict: The answer plus latency_s (seconds for the successful try) and attempts (number of tries used).

    Raises:
        Exception: The error from the last try when every try fails.
    """
    attempt = 1
    while True:
        start = time.perf_counter()
        try:
            if isinstance(client, AsyncTypeSafeClient):
                out = await call_jev(client, request)
            else:
                out = await call_openai(client, request)
            return out | {"latency_s": time.perf_counter() - start, "attempts": attempt}
        except Exception:
            if attempt == ATTEMPTS:
                raise
        await asyncio.sleep(2**attempt)
        attempt += 1


async def collect(
    api: ApiModel, ds: Dataset, X: pd.DataFrame, variant: str, cache_path: Path
) -> list[dict]:
    """Get one API answer per row and call the API only for rows not already cached.

    New answers are appended to cache_path after every CHUNK requests.

    Args:
        api (ApiModel): Model to call.
        ds (Dataset): Dataset the rows belong to.
        X (pd.DataFrame): Rows to answer.
        variant (str): Prompt variant.
        cache_path (Path): JSONL file of saved answers for this model and dataset.

    Returns:
        list[dict]: One answer per row of X in row order.

    Raises:
        RuntimeError: Some calls still failed after all retries. The other answers are saved,
            so a rerun fetches only the failed rows.
    """
    requests = {
        idx: build_request(api, zero_shot_task(ds, X.loc[idx], variant))
        for idx in X.index
    }
    hashes = {idx: request_hash(req) for idx, req in requests.items()}
    cache: dict[str, dict] = {}
    text = cache_path.read_text() if cache_path.exists() else ""
    for line in text.splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        cache[rec["hash"]] = rec

    # Rows with identical requests share one API call.
    first_row = {}
    for idx in X.index:
        first_row.setdefault(hashes[idx], idx)
    missing = [idx for h, idx in first_row.items() if h not in cache]
    if missing:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if text and not text.endswith("\n"):
            # An interrupted run may have left a partial last line, so new answers start on a fresh line.
            with cache_path.open("a") as f:
                f.write("\n")
        sem = asyncio.Semaphore(CONCURRENCY)
        # Aim for 90% of the rate limit.
        interval = 60 / (0.9 * api.requests_per_minute)
        failures = []
        async with make_client(api) as client:

            async def one(idx: int, position: int) -> dict:
                """Wait until position x interval seconds after the chunk starts and then send the request for the row whose index is idx."""
                # Space out request starts to stay under the rate limit, so retries stay rare.
                await asyncio.sleep(position * interval)
                async with sem:
                    out = await send(client, requests[idx])
                return {
                    "hash": hashes[idx],
                    "idx": str(idx),
                    "request": requests[idx],
                    **out,
                }

            # Save answers after each chunk, so an interrupted run keeps them and no write happens during timed requests.
            for i in range(0, len(missing), CHUNK):
                results = await asyncio.gather(
                    *(one(idx, k) for k, idx in enumerate(missing[i : i + CHUNK])),
                    return_exceptions=True,
                )
                with cache_path.open("a") as f:
                    for rec in results:
                        if isinstance(rec, BaseException):
                            failures.append(rec)
                        else:
                            f.write(json.dumps(rec) + "\n")
                            cache[rec["hash"]] = rec
        if failures:
            raise RuntimeError(
                f"{len(failures)} {api.name} calls failed; the rest are cached, rerun to fetch them"
            ) from failures[0]
    print(
        f"  {api.name}, {variant}: {len(missing)} API calls for {len(X)} rows",
        flush=True,
    )
    return [cache[hashes[idx]] for idx in X.index]


def parse_outputs(
    ds: Dataset, records: list[dict], index: pd.Index
) -> tuple[pd.Series, pd.DataFrame | None]:
    """Turn API answers into predicted labels and class probabilities.

    Labels are always one of ds.options, because both APIs restrict answers to the given options.
    JEV's probabilities are rescaled so that each row sums to exactly 1.

    Args:
        ds (Dataset): Dataset the answers belong to. Used for its labels.
        records (list[dict]): Answers from collect.
        index (pd.Index): Row index for the results.

    Returns:
        tuple[pd.Series, pd.DataFrame | None]: Predicted labels and the probability of each class for each row.
            Probabilities are None for OpenAI.
    """
    classes = sorted(ds.options)
    pred = pd.Series([r["label"] for r in records], index=index)
    if records[0]["probabilities"] is None:
        return pred, None
    proba = pd.DataFrame(
        [[r["probabilities"][c] for c in classes] for r in records],
        index=index,
        columns=classes,
    )
    # JEV's docs say its probabilities sum to approximately 1.
    return pred, proba.div(proba.sum(axis=1), axis=0)


# === Hugging Face Models ===

BART = "BART-MNLI (zero-shot)"
MODERNBERT = "ModernBERT (fine-tuned)"
BART_CHECKPOINT = "facebook/bart-large-mnli"
MODERNBERT_CHECKPOINT = "answerdotai/ModernBERT-base"


def bart_premise(task: ZeroShotTask) -> str:
    """The text BART judges. It is the question followed by the row text."""
    return f"{task.question}\n{task.text}"


def bart_hypotheses(task: ZeroShotTask, classes: list[str]) -> list[str]:
    """The statements BART checks the text against. There is one option description per class in the order of classes."""
    return [task.options[c] for c in classes]


def bart_zero_shot(
    ds: Dataset, X_te: pd.DataFrame, classes: list[str], variant: str
) -> Predictions:
    """Classify each row with BART-MNLI without training.

    Based on https://huggingface.co/facebook/bart-large-mnli

    Args:
        ds (Dataset): Dataset the rows belong to.
        X_te (pd.DataFrame): Rows to classify.
        classes (list[str]): Labels in the order of the probability columns.
        variant (str): Prompt variant.

    Returns:
        Predictions: Labels, probabilities and the time to predict one row. Training time is 0.
    """
    classifier = pipeline("zero-shot-classification", model=BART_CHECKPOINT)
    tasks = [zero_shot_task(ds, row, variant) for _, row in X_te.iterrows()]
    options = bart_hypotheses(tasks[0], classes)

    def to_proba(outputs: list[dict]) -> pd.DataFrame:
        """Put BART's per-option scores into one probability column per class in the order of classes."""
        by_option = pd.DataFrame([dict(zip(o["labels"], o["scores"])) for o in outputs])
        return by_option[options].set_axis(classes, axis=1)

    premises = [bart_premise(task) for task in tasks]
    # Use each option text as the hypothesis instead of the default "This example is {}.".
    proba = to_proba(
        classifier(premises, options, hypothesis_template="{}", batch_size=16)
    ).set_axis(X_te.index)
    latency = time_single_row(
        lambda row: classifier(
            bart_premise(zero_shot_task(ds, row.iloc[0], variant)),
            options,
            hypothesis_template="{}",
        ),
        X_te,
    )
    return Predictions(proba.idxmax(axis=1), proba, None, 0.0, latency)


def modernbert_finetune(
    ds: Dataset,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_te: pd.DataFrame,
    classes: list[str],
) -> Predictions:
    """Fine-tune ModernBERT on the training rows and then predict the test rows.

    10% of the training rows are held out to pick the best epoch. Training stops when validation loss stops
    improving or after 10 epochs.
    Based on https://github.com/AnswerDotAI/ModernBERT/blob/main/examples/finetune_modernbert_on_glue.ipynb
    The 10 epoch cap and early stopping follow https://arxiv.org/abs/2412.13663 (Appendix E.1).

    Args:
        ds (Dataset): Dataset the rows belong to.
        X_train (pd.DataFrame): Training rows.
        y_train (pd.Series): Training labels.
        X_te (pd.DataFrame): Rows to predict.
        classes (list[str]): Labels in the order of the probability columns.

    Returns:
        Predictions: Labels, probabilities and timings.
    """
    set_seed(SEED)
    label2id = {c: i for i, c in enumerate(classes)}
    tokenizer = AutoTokenizer.from_pretrained(MODERNBERT_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODERNBERT_CHECKPOINT,
        num_labels=len(classes),
        id2label=dict(enumerate(classes)),
        label2id=label2id,
    )

    def to_dataset(X: pd.DataFrame, y: pd.Series | None = None) -> HFDataset:
        """Turn rows into a tokenized Hugging Face dataset of row text in the list layout. Labels are added as numbers when y is given."""
        data: dict[str, list] = {
            "text": [row_text(ds, row, "list") for _, row in X.iterrows()]
        }
        if y is not None:
            data["label"] = [label2id[v] for v in y]
        return HFDataset.from_dict(data).map(
            lambda e: tokenizer(e["text"], truncation=True), batched=True
        )

    checkpoints = CACHE_DIR / f"modernbert_{ds.name}"
    args = TrainingArguments(
        output_dir=str(checkpoints),
        learning_rate=8e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=True,
        bf16_full_eval=True,
        report_to="none",
        seed=SEED,
    )
    X_fit, X_val, y_fit, y_val = holdout(
        X_train, y_train, None if ds.group is None else ds.group.loc[X_train.index]
    )
    start = time.perf_counter()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=to_dataset(X_fit, y_fit),
        eval_dataset=to_dataset(X_val, y_val),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=[EarlyStoppingCallback()],
    )
    trainer.train()
    train_time = time.perf_counter() - start
    shutil.rmtree(checkpoints)

    logits = trainer.predict(to_dataset(X_te)).predictions
    proba = pd.DataFrame(softmax(logits, axis=1), index=X_te.index, columns=classes)

    model.eval()

    def predict_row(row: pd.DataFrame) -> int:
        """Predict one row with the fine-tuned ModernBERT. The returned class index is ignored because only the time matters."""
        inputs = tokenizer(
            row_text(ds, row.iloc[0], "list"), return_tensors="pt", truncation=True
        ).to(model.device)
        with torch.no_grad():
            return model(**inputs).logits.argmax().item()

    return Predictions(
        proba.idxmax(axis=1),
        proba,
        None,
        train_time,
        time_single_row(predict_row, X_te),
    )


def truncated_rows(
    ds: Dataset, X_te: pd.DataFrame, classes: list[str]
) -> dict[str, int]:
    """Count the test rows whose text is longer than a model's token limit.

    ModernBERT reads the row text in the list layout and has a limit of 8,192 tokens.
    BART reads the question and row text together with the dataset's longest option description
    and has a limit of 1,024 tokens. BART is checked in every prompt variant and the counts are added up.
    Only the model groups listed in MODELS are checked.

    Args:
        ds (Dataset): Dataset the rows belong to.
        X_te (pd.DataFrame): Test rows.
        classes (list[str]): Labels.

    Returns:
        dict[str, int]: One count per checked model keyed by model name.
    """
    counts = {}
    if "modernbert" in MODELS:
        modern = AutoTokenizer.from_pretrained(MODERNBERT_CHECKPOINT)
        texts = [row_text(ds, row, "list") for _, row in X_te.iterrows()]
        counts[MODERNBERT] = sum(
            len(modern(t)["input_ids"]) > modern.model_max_length for t in texts
        )
    if "bart" in MODELS:
        bart = AutoTokenizer.from_pretrained(BART_CHECKPOINT)
        options = bart_hypotheses(zero_shot_task(ds, X_te.iloc[0], "list"), classes)
        longest_option = max(options, key=lambda o: len(bart(o)["input_ids"]))
        counts[BART] = sum(
            len(
                bart(bart_premise(zero_shot_task(ds, row, variant)), longest_option)[
                    "input_ids"
                ]
            )
            > bart.model_max_length
            for variant in VARIANTS
            for _, row in X_te.iterrows()
        )
    return counts


# === Evaluation ===


METRIC_COLS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc",
    "log_loss",
    "mcc",
    "train_time_s",
    "latency_ms",
]


def append_csv(path: Path, rows: pd.DataFrame) -> None:
    """Append rows to a CSV file and write the header only when the file is new."""
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)


def roc_auc(y_true: pd.Series, scores: pd.DataFrame) -> float:
    """Compute ROC-AUC from class scores.

    Binary tasks use the second column of scores, which is the second label in sorted order.
    Tasks with more classes average the AUC of each class against all the others.

    Args:
        y_true (pd.Series): True labels.
        scores (pd.DataFrame): Probability or SVM score of each class for each row.

    Returns:
        float: ROC-AUC.
    """
    classes = list(scores.columns)
    if len(classes) == 2:
        return roc_auc_score(y_true == classes[1], scores[classes[1]])
    return float(np.mean([roc_auc_score(y_true == c, scores[c]) for c in classes]))


def evaluate(
    y_true: pd.Series,
    pred: pd.Series,
    proba: pd.DataFrame | None,
    scores: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute every metric for one model.

    Args:
        y_true (pd.Series): True labels.
        pred (pd.Series): Predicted labels.
        proba (pd.DataFrame | None): Class probabilities. None when the model gives none.
        scores (pd.DataFrame | None): SVM score of each class. None for other models.

    Returns:
        dict[str, float]: Accuracy, macro precision, recall and F1, Matthews correlation coefficient (MCC),
            ROC-AUC and log-loss. ROC-AUC needs proba or scores. Log-loss needs proba. Missing metrics are NaN.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, pred),
        "precision_macro": precision_score(
            y_true, pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, pred),
        "roc_auc": np.nan,
        "log_loss": np.nan,
    }
    if proba is not None:
        metrics["roc_auc"] = roc_auc(y_true, proba)
        metrics["log_loss"] = log_loss(
            y_true, proba.to_numpy(), labels=list(proba.columns)
        )
    elif scores is not None:
        metrics["roc_auc"] = roc_auc(y_true, scores)
    return metrics


def prediction_rows(
    ds: Dataset,
    model: str,
    y_te: pd.Series,
    pred: pd.Series,
    proba: pd.DataFrame | None,
    records: list[dict] | None = None,
) -> pd.DataFrame:
    """Build the predictions.csv rows for one model.

    Args:
        ds (Dataset): Dataset the rows belong to.
        model (str): Model display name.
        y_te (pd.Series): True labels.
        pred (pd.Series): Predicted labels.
        proba (pd.DataFrame | None): Class probabilities or None.
        records (list[dict] | None): API answers for the token counts. None for local models.

    Returns:
        pd.DataFrame: One row per test row with the true and predicted label, probabilities and token counts.
    """
    rows = pd.DataFrame(
        {
            "dataset": ds.name,
            "model": model,
            "idx": y_te.index.astype(str),
            "y_true": y_te.to_numpy(),
            "y_pred": pred.to_numpy(),
            "proba": (
                [json.dumps(p) for p in proba.to_dict("records")]
                if proba is not None
                else ""
            ),
        }
    )
    for field in ["input_tokens", "output_tokens"]:
        rows[field] = [r[field] for r in records] if records else None
    return rows


# === Benchmark ===


def fit_predict(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_te: pd.DataFrame,
    classes: list[str],
) -> Predictions:
    """Train one sklearn pipeline and then predict the test rows.

    Args:
        pipe (Pipeline): Untrained pipeline from make_classifiers.
        X_train (pd.DataFrame): Training rows.
        y_train (pd.Series): Training labels.
        X_te (pd.DataFrame): Rows to predict.
        classes (list[str]): Labels in the order of the probability and score columns.

    Returns:
        Predictions: Labels, probabilities or SVM scores, and timings.
    """
    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    pred = pd.Series(pipe.predict(X_te), index=X_te.index)
    order = [list(pipe.classes_).index(c) for c in classes]
    proba = scores = None
    if hasattr(pipe, "predict_proba"):
        proba = pd.DataFrame(
            pipe.predict_proba(X_te)[:, order], index=X_te.index, columns=classes
        )
    else:
        raw = pipe.decision_function(X_te)
        # A binary SVM returns one score for the second class.
        if raw.ndim == 1:
            raw = np.column_stack([-raw, raw])
        scores = pd.DataFrame(raw[:, order], index=X_te.index, columns=classes)
    return Predictions(
        pred, proba, scores, train_time, time_single_row(pipe.predict, X_te)
    )


def run_dataset(ds: Dataset) -> dict:
    """Run every model group listed in MODELS on one dataset.

    The model groups are "api" (JEV and the OpenAI models), "sklearn", "bart" and "modernbert".
    API models and BART run once per prompt variant. Trained models run once per fold.
    As each model finishes, its metrics are appended to results.csv and its per-row predictions to predictions.csv.

    Args:
        ds (Dataset): Dataset to run.

    Returns:
        dict: This dataset's entry for run.json. It holds fold and row counts, a hash of the test rows and truncation counts.
    """
    folds = make_folds(ds)
    test_rows = [idx for _, te in folds for idx in te]
    X_te, y_te = ds.X.loc[test_rows], ds.y.loc[test_rows]
    print(f"\n{ds.name}: folds={len(folds)} test={len(X_te)}", flush=True)
    classes = sorted(ds.options)
    results_csv = RESULTS_DIR / "results.csv"
    preds_csv = RESULTS_DIR / "predictions.csv"

    def record(
        model: str, variant: str, metrics: dict[str, float], rows: pd.DataFrame
    ) -> None:
        """Append one model's metrics to results.csv and its per-row predictions to predictions.csv."""
        result = {"dataset": ds.name, "model": model, "variant": variant, **metrics}
        append_csv(
            results_csv,
            pd.DataFrame([result])[["dataset", "model", "variant"] + METRIC_COLS],
        )
        append_csv(preds_csv, rows.assign(variant=variant))

    if "api" in MODELS:
        for variant in VARIANTS:
            for api in API_MODELS:
                records = asyncio.run(
                    collect(
                        api, ds, X_te, variant, CACHE_DIR / f"{api.key}_{ds.name}.jsonl"
                    )
                )
                pred, proba = parse_outputs(ds, records, X_te.index)
                latency = float(np.median([r["latency_s"] for r in records]))
                metrics = evaluate(y_te, pred, proba) | {
                    "train_time_s": 0.0,
                    "latency_ms": latency * 1000,
                }
                record(
                    api.name,
                    variant,
                    metrics,
                    prediction_rows(ds, api.name, y_te, pred, proba, records),
                )

    def score(name: str, out: Predictions, variant: str = "") -> None:
        """Compute the metrics of one model's predictions and record them. variant is empty for trained models."""
        metrics = evaluate(y_te, out.labels, out.proba, out.scores) | {
            "train_time_s": out.train_seconds,
            "latency_ms": out.row_seconds * 1000,
        }
        record(
            name,
            variant,
            metrics,
            prediction_rows(ds, name, y_te, out.labels, out.proba),
        )
        label = f"{name}, {variant}" if variant else name
        print(f"  {label}: done, {out.train_seconds:.1f}s training", flush=True)

    def train_on_folds(
        name: str,
        fit_predict_fold: Callable[
            [pd.DataFrame, pd.Series, pd.DataFrame], Predictions
        ],
    ) -> None:
        """Train once per fold, join the predictions from all folds, and score them.

        Training time and the time to predict one row are both the mean over folds.
        """
        outs = [
            fit_predict_fold(ds.X.loc[tr], ds.y.loc[tr], ds.X.loc[te])
            for tr, te in folds
        ]
        join = lambda parts: None if parts[0] is None else pd.concat(parts)
        score(
            name,
            Predictions(
                pd.concat([o.labels for o in outs]),
                join([o.proba for o in outs]),
                join([o.scores for o in outs]),
                float(np.mean([o.train_seconds for o in outs])),
                float(np.mean([o.row_seconds for o in outs])),
            ),
        )

    if "sklearn" in MODELS:
        for name, pipe in make_classifiers(ds.X).items():
            train_on_folds(
                name, lambda Xtr, ytr, Xte: fit_predict(pipe, Xtr, ytr, Xte, classes)
            )

    truncated = truncated_rows(ds, X_te, classes)
    if any(truncated.values()):
        raise RuntimeError(
            f"{ds.name}: some test rows exceed a model's token limit: {truncated}"
        )
    if "bart" in MODELS:
        for variant in VARIANTS:
            score(BART, bart_zero_shot(ds, X_te, classes, variant), variant)
    if "modernbert" in MODELS:
        train_on_folds(
            MODERNBERT,
            lambda Xtr, ytr, Xte: modernbert_finetune(ds, Xtr, ytr, Xte, classes),
        )

    test_ids = ",".join(sorted(X_te.index.astype(str)))
    return {
        "n_folds": len(folds),
        "n_train": [len(tr) for tr, _ in folds],
        "n_test": len(X_te),
        "test_rows_sha256": hashlib.sha256(test_ids.encode()).hexdigest(),
        "majority_share": float(y_te.value_counts(normalize=True).max()),
        "truncated_test_rows": truncated,
    }


def cpu_name() -> str:
    """Return the CPU model name.

    Reads /proc/cpuinfo on Linux and falls back to platform.processor() elsewhere.

    Returns:
        str: CPU model name.
    """
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def main() -> None:
    """Run MODELS on DATASETS and save the results.

    Deletes any existing results.csv, predictions.csv and run.json in RESULTS_DIR first.
    """
    load_dotenv(Path(__file__).parent / ".env")
    RESULTS_DIR.mkdir(exist_ok=True)
    for name in ["results.csv", "predictions.csv", "run.json"]:
        (RESULTS_DIR / name).unlink(missing_ok=True)
    manifest = {
        "started": datetime.now(UTC).isoformat(),
        "datasets_run": DATASETS,
        "models_run": MODELS,
        "variants": VARIANTS,
        "seed": SEED,
        "test_size": 0.1,
        "kfold_below_rows": KFOLD_BELOW_ROWS,
        "concurrency": CONCURRENCY,
        "attempts": ATTEMPTS,
        "api_models": [asdict(a) for a in API_MODELS],
        "hf_checkpoints": {BART: BART_CHECKPOINT, MODERNBERT: MODERNBERT_CHECKPOINT},
        "versions": {
            p: metadata.version(p)
            for p in [
                "scikit-learn",
                "openai",
                "typesafe-sdk",
                "transformers",
                "torch",
                "datasets",
                "accelerate",
            ]
        },
        "cpu": cpu_name(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else None
        ),
        "datasets": {},
    }
    loaders = {
        "diabetes": load_diabetes,
        "titanic": load_titanic,
        "wine": load_wine_cultivars,
        "diabetes130": load_diabetes130,
        "adult": load_adult,
    }
    for name in DATASETS:
        ds = loaders[name]()
        manifest["datasets"][ds.name] = run_dataset(ds)
        (RESULTS_DIR / "run.json").write_text(json.dumps(manifest, indent=2))
    manifest["finished"] = datetime.now(UTC).isoformat()
    (RESULTS_DIR / "run.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved results.csv, predictions.csv, run.json in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
