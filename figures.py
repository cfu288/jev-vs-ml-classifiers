"""Make the charts and summary.png from the files in results/."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.ticker import FixedLocator, NullFormatter

from compare import API_MODELS, BART, RESULTS_DIR

# === Style ===

PACK = "#898781"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
API_COLORS = dict(
    zip([a.name for a in API_MODELS], ["#2a78d6", "#eb6834", "#4a3aa7", "#1baf7a"])
)
ZERO_SHOT_COLORS = API_COLORS | {BART: "#b8325e"}
BASELINE = "Majority class"
BELOW_BASELINE = "#d6d5cf"


def display_name(model: str) -> str:
    """Chart label for a model.

    The baseline becomes "Always guess the most common class". " (zero-shot)" is removed from every name.
    OpenAI models then get their reasoning effort appended as ", minimal reasoning" or ", no reasoning".
    Other names are unchanged.

    Args:
        model (str): Model name as stored in results.csv.

    Returns:
        str: Label for charts.
    """
    if model == BASELINE:
        return "Always guess the most common class"
    base = model.replace(" (zero-shot)", "")
    api = next((a for a in API_MODELS if a.name == model), None)
    if api is None or api.effort is None:
        return base
    effort = {"none": "no reasoning", "minimal": "minimal reasoning"}[api.effort]
    return f"{base}, {effort}"


TASKS = {
    "diabetes": "Diabetes onset",
    "titanic": "Titanic survival",
    "wine": "Wine class",
    "diabetes130": "Readmission",
    "adult": "Census income",
}

# Dollars per single input token and per single output token. OpenAI list prices from developers.openai.com/api/docs/pricing on 2026-09-22. JEV charges only for input during early access according to docs.typesafe.ai/models.
RATES = {
    "JEV (zero-shot)": (42 / 1e9, 0.0),
    "GPT-5-nano (zero-shot)": (0.05 / 1e6, 0.40 / 1e6),
    "GPT-5-mini (zero-shot)": (0.25 / 1e6, 2.00 / 1e6),
    "GPT-6-luna (zero-shot)": (0.10 / 1e6, 0.50 / 1e6),
}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "font.size": 11,
    }
)


# === Data ===


def load_run() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load results.csv, predictions.csv and run.json.

    Zero-shot scores are averaged over the prompt variants.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]: Metrics with one row per model and dataset, per-row predictions,
            and the contents of run.json.

    Raises:
        RuntimeError: A model is missing a dataset.
    """
    df = (
        pd.read_csv(RESULTS_DIR / "results.csv")
        .groupby(["dataset", "model"], as_index=False)
        .mean(numeric_only=True)
    )
    preds = pd.read_csv(RESULTS_DIR / "predictions.csv", low_memory=False)
    run = json.loads((RESULTS_DIR / "run.json").read_text())
    counts = df.groupby(["model", "dataset"]).size().unstack()
    if set(counts.columns) != set(run["datasets"]) or counts.isna().any().any():
        raise RuntimeError(f"results.csv is not one complete run:\n{counts}")
    return df, preds, run


def zero_shot(df: pd.DataFrame) -> list[str]:
    """List the zero-shot models present in the results.

    Args:
        df (pd.DataFrame): Metrics from load_run.

    Returns:
        list[str]: Zero-shot model names.
    """
    return [m for m in ZERO_SHOT_COLORS if m in set(df.model)]


def model_order(df: pd.DataFrame) -> list[str]:
    """Order the models for the charts.

    Trained models come first and zero-shot models second. Each group is sorted from highest to lowest average macro-F1.
    The majority-class baseline is left out.

    Args:
        df (pd.DataFrame): Metrics from load_run.

    Returns:
        list[str]: Model names in chart order.
    """
    avg = df.groupby("model").f1_macro.mean()
    trained = (
        avg.drop(index=zero_shot(df) + [BASELINE])
        .sort_values(ascending=False)
        .index.tolist()
    )
    return trained + avg.loc[zero_shot(df)].sort_values(ascending=False).index.tolist()


# === Figures ===


def style(ax: Axes) -> None:
    """Hide the top, left and right borders and the tick marks."""
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def fig_latency(df: pd.DataFrame) -> None:
    """Save results/fig2_latency.png with the median time per prediction for each model.

    Each model's time is the median over the datasets.

    Args:
        df (pd.DataFrame): Metrics from load_run.
    """
    order = model_order(df)
    lat = df.pivot(index="model", columns="dataset", values="latency_ms").median(
        axis=1
    )[order]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for y, model in enumerate(order):
        if model in ZERO_SHOT_COLORS:
            ax.scatter(lat[model], y, s=120, color=ZERO_SHOT_COLORS[model], zorder=4)
        else:
            ax.scatter(
                lat[model],
                y,
                s=70,
                facecolors="white",
                edgecolors=PACK,
                linewidths=1.8,
                zorder=3,
            )
        ax.text(
            lat[model] * 1.35,
            y,
            f"{lat[model]:,.0f} ms" if lat[model] >= 10 else f"{lat[model]:.1f} ms",
            va="center",
            fontsize=10,
            color=INK2,
        )
    ax.axhline(len(order) - len(zero_shot(df)) - 0.5, color=GRID, linewidth=1)
    ax.set_yticks(range(len(order)), [display_name(m) for m in order], fontsize=11)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xscale("log")
    ax.set_xlim(lat.min() * 0.5, lat.max() * 2.6)
    majors = [
        t
        for t in [0.1, 1, 10, 100, 1000, 10000]
        if lat.min() * 0.5 <= t <= lat.max() * 2.6
    ]
    ax.set_xticks(
        majors,
        [f"{t:g} ms" if t < 1000 else f"{t / 1000:g} s" for t in majors],
        fontsize=11,
    )
    ax.set_xlabel(
        "Median time per prediction (log scale; faint lines every 100 ms)", fontsize=12
    )
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(100, lat.max() * 2.6, 100)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", which="major", color="#b8b7b0", linewidth=1.0)
    ax.grid(axis="x", which="minor", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    style(ax)
    ax.set_title(
        "Time per prediction (API: one request; local: one-row predict)",
        loc="left",
        fontsize=13,
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig2_latency.png", dpi=150)
    plt.close(fig)


def cost_per_1k(preds: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Estimate each API model's cost per 1,000 predictions at list prices.

    Args:
        preds (pd.DataFrame): Per-row predictions from load_run with token counts.

    Returns:
        dict[str, dict[str, float]]: For each API model, {"input": dollars, "output": dollars} per 1,000 predictions.

    Raises:
        RuntimeError: An API model has no rows or is missing token counts.
    """
    out = {}
    for model, (in_rate, out_rate) in RATES.items():
        rows = preds[preds.model == model]
        if rows.empty:
            raise RuntimeError(f"{model} has no prediction rows")
        if rows[["input_tokens", "output_tokens"]].isna().any().any():
            raise RuntimeError(f"{model} rows are missing token counts")
        out[model] = {
            "input": rows.input_tokens.sum() * in_rate / len(rows) * 1000,
            "output": rows.output_tokens.sum() * out_rate / len(rows) * 1000,
        }
    return out


def fig_cost(preds: pd.DataFrame) -> None:
    """Save results/fig3_cost.png with cost bars from cheapest to most expensive.

    Args:
        preds (pd.DataFrame): Per-row predictions from load_run with token counts.
    """
    costs = cost_per_1k(preds)
    fig, ax = plt.subplots(figsize=(9, 3.9))
    names = sorted(costs, key=lambda m: costs[m]["input"] + costs[m]["output"])
    totals = [costs[m]["input"] + costs[m]["output"] for m in names]
    for y, model in enumerate(reversed(names)):
        c = costs[model]
        ax.barh(y, c["input"], height=0.5, color=API_COLORS[model])
        ax.barh(
            y,
            c["output"],
            left=c["input"],
            height=0.5,
            color=API_COLORS[model],
            alpha=0.45,
        )
        out_text = (
            "output uncharged (early access)"
            if RATES[model][1] == 0
            else f"output \\${c['output']:.3f}"
        )
        ax.text(
            c["input"] + c["output"] + max(totals) * 0.02,
            y,
            f"\\${c['input'] + c['output']:.3f}  (input \\${c['input']:.3f} + {out_text})",
            va="center",
            fontsize=10,
            color=INK,
        )
    ax.set_yticks(range(len(names)), [display_name(m) for m in reversed(names)])
    ax.set_xlim(0, max(totals) * 1.9)
    ax.set_xlabel(
        "Estimated \\$ per 1,000 classifications at list prices (local models: ≈ \\$0)"
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style(ax)
    ax.set_title(
        "Estimated cost per 1,000 classifications", loc="left", fontsize=13, pad=14
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig3_cost.png", dpi=150)
    plt.close(fig)


def fig_f1_heatmap(df: pd.DataFrame, run: dict) -> None:
    """Save the macro-F1 table to results/fig1_f1_heatmap.png.

    Each column is colored from its lowest to highest score.
    Cells whose score rounded to two decimals is at or below the majority-class baseline's are grey.

    Args:
        df (pd.DataFrame): Metrics from load_run.
        run (dict): Contents of run.json with the dataset sizes and fold counts.
    """
    order = model_order(df)
    zero_shot_models = zero_shot(df)
    trained = [m for m in order if m not in zero_shot_models]
    sections = [
        ("Trained on each dataset", trained),
        (
            "Zero-shot, averaged over prompt wordings",
            [m for m in order if m in zero_shot_models],
        ),
        ("Majority-class baseline", [BASELINE]),
    ]
    rows = [row for title, models in sections for row in [title] + models]
    datasets = list(run["datasets"])
    f1 = df.pivot(index="model", columns="dataset", values="f1_macro").reindex(rows)[
        datasets
    ]
    f1["average"] = f1.mean(axis=1)
    headers = [
        f"{TASKS[d]}\n(n = {run['datasets'][d]['n_test']:,}"
        + (", 10-fold)" if run["datasets"][d]["n_folds"] > 1 else ")")
        for d in datasets
    ] + ["Average\n(equal weight)"]
    fig, ax = plt.subplots(figsize=(11.5, 8.6))
    vals = f1.to_numpy(dtype=float)
    floor = f1.loc[BASELINE].to_numpy(dtype=float)
    lo, hi = f1.loc[order].min().to_numpy(), f1.loc[order].max().to_numpy()
    shade = (vals - lo) / np.where(hi > lo, hi - lo, 1)
    grey = vals.round(2) <= floor.round(2)
    colors = plt.get_cmap("RdYlBu")(np.clip(shade, 0, 1))
    colors[grey] = to_rgba(BELOW_BASELINE)
    is_title = f1.index.isin([title for title, _ in sections])
    colors[is_title] = (0, 0, 0, 0)
    ax.imshow(colors, aspect="auto")
    for i in range(vals.shape[0]):
        if is_title[i]:
            ax.text(
                -0.45,
                i + 0.2,
                f1.index[i],
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=INK,
            )
            continue
        for j in range(vals.shape[1]):
            # White text on the darkest ends of the color scale.
            dark = not grey[i, j] and (shade[i, j] > 0.85 or shade[i, j] < 0.12)
            ax.text(
                j,
                i,
                f"{vals[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9.5,
                color="white" if dark else INK,
            )
    ax.set_xticks(range(len(headers)), headers, fontsize=9.5)
    ax.set_yticks(
        range(len(f1)),
        ["" if t else display_name(m) for m, t in zip(f1.index, is_title)],
        fontsize=10,
    )
    ax.set_xticks(np.arange(-0.5, len(headers)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(f1)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    title_rows = np.flatnonzero(is_title).tolist() + [len(f1)]
    for start, stop in zip(title_rows, title_rows[1:]):
        ax.vlines(len(datasets) - 0.5, start + 0.5, stop - 0.5, color=INK, linewidth=2)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(which="both", length=0)
    ax.xaxis.tick_top()
    ax.set_title("Macro-F1 by model and dataset", loc="left", fontsize=13, pad=62)
    ax.annotate(
        "Blue = high, red = low, grey = at or below the majority-class baseline",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 42),
        textcoords="offset points",
        fontsize=10,
        color=INK2,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig1_f1_heatmap.png", dpi=150)
    plt.close(fig)


def summary_image() -> None:
    """Stack the heatmap, the latency and cost charts, and the boundary mosaic if it exists into results/summary.png."""
    load = lambda name: plt.imread(RESULTS_DIR / name)
    heat, lat, cost = (
        load("fig1_f1_heatmap.png"),
        load("fig2_latency.png"),
        load("fig3_cost.png"),
    )
    bound_path = RESULTS_DIR / "fig4_boundaries.png"
    bound = plt.imread(bound_path) if bound_path.exists() else None
    width = max(
        img.shape[1]
        for img in [heat, lat, cost] + ([bound] if bound is not None else [])
    )
    gap = 40

    def height_at(img: np.ndarray, w: float) -> float:
        """Height of img when scaled to width w."""
        return img.shape[0] * w / img.shape[1]

    heat_h = height_at(heat, width)
    pair_h = width / (lat.shape[1] / lat.shape[0] + cost.shape[1] / cost.shape[0])
    lat_w = pair_h * lat.shape[1] / lat.shape[0]
    bound_h = height_at(bound, width) if bound is not None else 0.0
    total_h = heat_h + gap + pair_h + (gap + bound_h if bound is not None else 0.0)

    fig = plt.figure(figsize=(width / 100, total_h / 100), dpi=100)

    def place(img: np.ndarray, x: float, y: float, w: float, h: float) -> None:
        """Draw img in the box at (x, y) with width w and height h. All sizes are in pixels of the summary image."""
        ax = fig.add_axes((x / width, 1 - (y + h) / total_h, w / width, h / total_h))
        ax.imshow(img, aspect="auto")
        ax.axis("off")

    place(heat, 0, 0, width, heat_h)
    place(lat, 0, heat_h + gap, lat_w, pair_h)
    place(cost, lat_w, heat_h + gap, width - lat_w, pair_h)
    if bound is not None:
        place(bound, 0, heat_h + gap + pair_h + gap, width, bound_h)
    fig.savefig(RESULTS_DIR / "summary.png", dpi=100, facecolor="white")
    plt.close(fig)
    print(
        "wrote results/summary.png"
        + ("" if bound is not None else " (without the boundary mosaic)")
    )


def main() -> None:
    """Remake all charts and summary.png from the saved results."""
    df, preds, run = load_run()
    fig_latency(df)
    fig_cost(preds)
    fig_f1_heatmap(df, run)
    print("wrote results/fig1_f1_heatmap.png, fig2_latency.png, fig3_cost.png")
    summary_image()


if __name__ == "__main__":
    main()
