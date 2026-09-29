"""Every zero-shot model receives the same question, options and row text in each prompt variant."""

import pandas as pd
import pytest

from compare import (
    API_MODELS,
    Dataset,
    bart_hypotheses,
    bart_premise,
    build_request,
    zero_shot_task,
)

DATASET = Dataset(
    name="toy",
    X=pd.DataFrame({"age": [50.0], "smoker": ["yes"]}),
    y=pd.Series(["sick"]),
    readable={"age": "Age (years)", "smoker": "Smoker"},
    subject="A patient record.",
    questions={
        "list": "Will this patient get sick?",
        "short": "Sick?",
        "structured": "You are reviewing a patient record. Classify whether this patient will get sick.",
    },
    options={"sick": "The patient gets sick", "well": "The patient stays well"},
)
ROW = DATASET.X.iloc[0]
JEV, NANO = API_MODELS[0], API_MODELS[1]
SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["sick", "well"]}},
    "required": ["label"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    ("variant", "question", "row"),
    [
        (
            "list",
            "Will this patient get sick?",
            "A patient record.\n- Age (years): 50\n- Smoker: yes",
        ),
        ("short", "Sick?", "A patient record.\nAge (years): 50, Smoker: yes"),
        (
            "structured",
            "You are reviewing a patient record. Classify whether this patient will get sick.",
            '{"subject": "A patient record.", "features": {"Age (years)": 50, "Smoker": "yes"}}',
        ),
    ],
)
def test_jev_request_carries_question_options_and_row(variant, question, row):
    assert build_request(JEV, zero_shot_task(DATASET, ROW, variant)) == {
        "provider": "jev",
        "model": "jev-1.13.0",
        "instructions": question,
        "criteria": {"sick": "The patient gets sick", "well": "The patient stays well"},
        "state": row,
    }


@pytest.mark.parametrize(
    ("variant", "instructions", "row"),
    [
        (
            "list",
            (
                "Will this patient get sick?\n"
                "Answer with exactly one of the labels below.\n"
                "Options:\n"
                "- sick: The patient gets sick\n"
                "- well: The patient stays well"
            ),
            "A patient record.\n- Age (years): 50\n- Smoker: yes",
        ),
        (
            "short",
            (
                "Sick?\n"
                "Answer with exactly one of the labels below.\n"
                "Options:\n"
                "- sick: The patient gets sick\n"
                "- well: The patient stays well"
            ),
            "A patient record.\nAge (years): 50, Smoker: yes",
        ),
        (
            "structured",
            (
                "You are reviewing a patient record. Classify whether this patient will get sick.\n"
                "Answer with exactly one of the labels below.\n"
                "Options:\n"
                "- sick: The patient gets sick\n"
                "- well: The patient stays well"
            ),
            '{"subject": "A patient record.", "features": {"Age (years)": 50, "Smoker": "yes"}}',
        ),
    ],
)
def test_openai_request_carries_question_options_and_row(variant, instructions, row):
    assert build_request(NANO, zero_shot_task(DATASET, ROW, variant)) == {
        "provider": "openai",
        "model": "gpt-5-nano-2025-08-07",
        "effort": "minimal",
        "instructions": instructions,
        "input": row,
        "schema": SCHEMA,
    }


@pytest.mark.parametrize(
    ("variant", "premise"),
    [
        (
            "list",
            "Will this patient get sick?\nA patient record.\n- Age (years): 50\n- Smoker: yes",
        ),
        ("short", "Sick?\nA patient record.\nAge (years): 50, Smoker: yes"),
        (
            "structured",
            (
                "You are reviewing a patient record. Classify whether this patient will get sick.\n"
                '{"subject": "A patient record.", "features": {"Age (years)": 50, "Smoker": "yes"}}'
            ),
        ),
    ],
)
def test_bart_premise_carries_question_and_row(variant, premise):
    assert bart_premise(zero_shot_task(DATASET, ROW, variant)) == premise


@pytest.mark.parametrize("variant", ["list", "short", "structured"])
def test_bart_hypotheses_are_the_options(variant):
    assert bart_hypotheses(zero_shot_task(DATASET, ROW, variant), ["sick", "well"]) == [
        "The patient gets sick",
        "The patient stays well",
    ]
