"""Draw the decision-boundary mosaic with one panel per model and dataset.

Every model sees only two features per dataset. The diabetes panels use glucose and BMI.
The wine panels use flavanoids and proline.
Each grid cell shows the label a model predicts there. API models make one call per grid cell.
Based on https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
API answers are cached in results/cache/boundary_*.jsonl.
"""

import asyncio
import json
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

from compare import (
    API_MODELS,
    CACHE_DIR,
    RESULTS_DIR,
    collect,
    load_diabetes,
    load_wine_cultivars,
    make_classifiers,
    parse_outputs,
    split_dataset,
)
from figures import display_name

# === Settings ===

SKLEARN_MODELS = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
]
VIEWS = [
    ("Diabetes onset", load_diabetes, ["plas", "mass"], ["Glucose", "BMI"]),
    (
        "Wine class",
        load_wine_cultivars,
        ["flavanoids", "proline"],
        ["Flavanoids", "Proline"],
    ),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
GRID_N = 50
PAD = 0.05


# === Drawing ===


def grid_points(X2: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build a GRID_N x GRID_N grid over two features with 5% padding on each side.

    Args:
        X2 (pd.DataFrame): The two features.

    Returns:
        tuple[np.ndarray, np.ndarray, pd.DataFrame]: Grid x and y coordinates for plotting and the grid points as rows.
    """
    lo, hi = X2.min(), X2.max()
    span = hi - lo
    xs = np.linspace(
        lo.iloc[0] - PAD * span.iloc[0], hi.iloc[0] + PAD * span.iloc[0], GRID_N
    )
    ys = np.linspace(
        lo.iloc[1] - PAD * span.iloc[1], hi.iloc[1] + PAD * span.iloc[1], GRID_N
    )
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy, pd.DataFrame({X2.columns[0]: xx.ravel(), X2.columns[1]: yy.ravel()})


def draw_labels(
    ax: Axes,
    xx: np.ndarray,
    yy: np.ndarray,
    classes: list[str],
    preds: pd.Series,
    cmap: ListedColormap,
) -> None:
    """Color each grid cell by the label predicted there.

    Args:
        ax (Axes): Panel to draw on.
        xx (np.ndarray): Grid x coordinates.
        yy (np.ndarray): Grid y coordinates.
        classes (list[str]): Labels in color order.
        preds (pd.Series): Predicted label for each grid point.
        cmap (ListedColormap): One color per label.
    """
    codes = (
        preds.map({c: i for i, c in enumerate(classes)}).to_numpy().reshape(xx.shape)
    )
    ax.pcolormesh(
        xx,
        yy,
        codes,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(classes) - 0.5,
        alpha=0.55,
        shading="nearest",
    )


def token_summary() -> None:
    """Print the call and token totals for each boundary cache file."""
    for f in sorted(CACHE_DIR.glob("boundary_*.jsonl")):
        recs = [json.loads(line) for line in f.read_text().splitlines()]
        tin = sum(r["input_tokens"] for r in recs)
        tout = sum(r["output_tokens"] for r in recs)
        print(f"{f.name}: {len(recs)} calls, {tin} input tokens, {tout} output tokens")


# === Main ===


def main() -> None:
    """Draw the mosaic to results/fig4_boundaries.png and then remake summary.png if the other charts exist."""
    load_dotenv(RESULTS_DIR.parent / ".env")
    cols = 1 + len(SKLEARN_MODELS) + len(API_MODELS)
    fig, axes = plt.subplots(len(VIEWS), cols, figsize=(2.6 * cols, 5.6))
    for r, (title, loader, pair, pair_labels) in enumerate(VIEWS):
        ds = loader()
        ds2 = replace(ds, X=ds.X[pair], readable={k: ds.readable[k] for k in pair})
        classes = sorted(ds.options)
        color = dict(zip(classes, COLORS))
        cmap = ListedColormap([color[c] for c in classes])
        X_tr, X_te, y_tr, y_te = split_dataset(ds2)
        xx, yy, pts = grid_points(ds2.X)

        def finish(ax: Axes, name: str) -> None:
            """Draw the training points and the faded test points, set the axis limits and title the panel.

            Args:
                ax (Axes): Panel to draw on.
                name (str): Model name for the panel title.
            """
            ax.scatter(
                X_tr[pair[0]],
                X_tr[pair[1]],
                c=y_tr.map(color),
                s=9,
                edgecolors="k",
                linewidths=0.3,
            )
            ax.scatter(
                X_te[pair[0]],
                X_te[pair[1]],
                c=y_te.map(color),
                s=9,
                edgecolors="k",
                linewidths=0.3,
                alpha=0.5,
            )
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if r == 0:
                ax.set_title(display_name(name).replace(", ", "\n"), fontsize=10)

        ax = axes[r, 0]
        ax.scatter(
            ds2.X[pair[0]],
            ds2.X[pair[1]],
            c=ds.y.map(color),
            s=11,
            edgecolors="k",
            linewidths=0.3,
        )
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_ylabel(f"{title}\n{pair_labels[1]}")
        ax.set_xlabel(pair_labels[0])
        if r == 0:
            ax.set_title("Input data", fontsize=10)

        classifiers = make_classifiers(ds2.X)
        for c, name in enumerate(SKLEARN_MODELS, start=1):
            pipe = classifiers[name]
            pipe.fit(X_tr, y_tr)
            draw_labels(axes[r, c], xx, yy, classes, pd.Series(pipe.predict(pts)), cmap)
            finish(axes[r, c], name)

        for c, api in enumerate(API_MODELS, start=1 + len(SKLEARN_MODELS)):
            records = asyncio.run(
                collect(
                    api,
                    ds2,
                    pts,
                    "list",
                    CACHE_DIR / f"boundary_{api.key}_{ds.name}.jsonl",
                )
            )
            preds, _ = parse_outputs(ds2, records, pts.index)
            draw_labels(axes[r, c], xx, yy, classes, preds, cmap)
            finish(axes[r, c], api.name)

    fig.suptitle("Predicted label on two features per dataset", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig4_boundaries.png", dpi=150)
    plt.close(fig)
    print("wrote results/fig4_boundaries.png")
    token_summary()
    if all(
        (RESULTS_DIR / name).exists()
        for name in ["fig1_f1_heatmap.png", "fig2_latency.png", "fig3_cost.png"]
    ):
        import figures

        figures.summary_image()


if __name__ == "__main__":
    main()
