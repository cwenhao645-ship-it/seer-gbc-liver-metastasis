"""Redraw final Reviewer 1 figures from locked source predictions and SHAP values."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


SEED = 20260721
MODEL_ORDER = ["Linear Logistic", "Spline Logistic", "XGBoost", "CatBoost"]
MODEL_COLORS = {
    "Linear Logistic": "#0072B2",
    "Spline Logistic": "#009E73",
    "XGBoost": "#E69F00",
    "CatBoost": "#CC79A7",
}
MODEL_STYLES = {
    "Linear Logistic": "-",
    "Spline Logistic": "--",
    "XGBoost": "-.",
    "CatBoost": ":",
}
RISK_ORDER = ["Low", "Intermediate", "High"]
RISK_COLORS = {"Low": "#0072B2", "Intermediate": "#E69F00", "High": "#D55E00"}
TEXT = "#222222"
GRID = "#D9D9D9"


def parse_args() -> argparse.Namespace:
    version_root = Path(__file__).resolve().parents[1]
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=project_root / "00_original_snapshot" / "test_set_predictions.csv",
    )
    parser.add_argument(
        "--shap",
        type=Path,
        default=project_root / "00_original_snapshot" / "SHAP_values_long_selected_model.csv",
    )
    parser.add_argument("--figures-dir", type=Path, default=version_root / "07_figures")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=version_root / "03_analysis_review" / "final_figure_source",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def configure_style() -> None:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = "Arial" if "Arial" in installed else "DejaVu Sans"
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.edgecolor": "#777777",
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.07,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def wilson_interval(events: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    proportion = events / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def calibration_points(y: np.ndarray, p: np.ndarray, bins: int = 6) -> pd.DataFrame:
    data = pd.DataFrame({"outcome": y, "probability": p})
    data["bin"] = pd.qcut(data["probability"], q=bins, duplicates="drop")
    rows = []
    for index, group in data.groupby("bin", observed=True):
        events = int(group["outcome"].sum())
        low, high = wilson_interval(events, len(group))
        rows.append(
            {
                "bin": str(index),
                "n": len(group),
                "events": events,
                "mean_predicted": group["probability"].mean(),
                "observed": events / len(group),
                "observed_ci_low": low,
                "observed_ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def decision_curve(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    n = len(y)
    for threshold in thresholds:
        predicted = p >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        weight = threshold / (1 - threshold)
        rows.append(
            {
                "threshold": threshold,
                "net_benefit": tp / n - fp / n * weight,
            }
        )
    return pd.DataFrame(rows)


def figure3(predictions: pd.DataFrame, outdir: Path, sourcedir: Path) -> None:
    data = predictions[predictions["Tier"] == "Tier3"].copy()
    models = [model for model in MODEL_ORDER if model in set(data["Model"])]
    prevalence = float(data[data["Model"] == models[0]]["y_true"].mean())
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.3), constrained_layout=True)
    ax_roc, ax_pr, ax_cal, ax_dca = axes.ravel()
    source_roc = []
    source_pr = []
    source_cal = []
    source_dca = []

    for model in models:
        subset = data[data["Model"] == model]
        y = subset["y_true"].to_numpy(dtype=int)
        p = subset["pred_prob"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        roc_value = roc_auc_score(y, p)
        pr_value = average_precision_score(y, p)
        ax_roc.plot(
            fpr,
            tpr,
            color=MODEL_COLORS[model],
            linestyle=MODEL_STYLES[model],
            linewidth=2.0,
            label=f"{model}  {roc_value:.3f}",
        )
        ax_pr.plot(
            recall,
            precision,
            color=MODEL_COLORS[model],
            linestyle=MODEL_STYLES[model],
            linewidth=2.0,
            label=f"{model}  {pr_value:.3f}",
        )
        source_roc.append(pd.DataFrame({"Model": model, "FPR": fpr, "TPR": tpr}))
        source_pr.append(pd.DataFrame({"Model": model, "Recall": recall, "Precision": precision}))

        calibration = calibration_points(y, p, bins=6)
        calibration.insert(0, "Model", model)
        source_cal.append(calibration)
        ax_cal.plot(
            calibration["mean_predicted"],
            calibration["observed"],
            color=MODEL_COLORS[model],
            linestyle=MODEL_STYLES[model],
            marker="o" if model == "CatBoost" else None,
            markersize=4.2,
            linewidth=2.0 if model == "CatBoost" else 1.4,
            alpha=1.0 if model == "CatBoost" else 0.80,
            label=model,
        )
        if model == "CatBoost":
            lower = calibration["observed"] - calibration["observed_ci_low"]
            upper = calibration["observed_ci_high"] - calibration["observed"]
            ax_cal.errorbar(
                calibration["mean_predicted"],
                calibration["observed"],
                yerr=np.vstack([lower, upper]),
                fmt="none",
                ecolor=MODEL_COLORS[model],
                elinewidth=1.0,
                capsize=2.5,
                alpha=0.9,
            )

        curve = decision_curve(y, p, np.linspace(0.01, 0.30, 100))
        curve.insert(0, "Model", model)
        source_dca.append(curve)
        ax_dca.plot(
            curve["threshold"],
            curve["net_benefit"],
            color=MODEL_COLORS[model],
            linestyle=MODEL_STYLES[model],
            linewidth=2.0,
            label=model,
        )

    ax_roc.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1.0)
    ax_roc.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax_roc.set_title("Receiver operating characteristic curves", loc="left", fontweight="bold")
    ax_roc.legend(title="Model  ROC-AUC", loc="lower right")

    ax_pr.axhline(prevalence, color="#888888", linestyle="--", linewidth=1.0, label=f"Prevalence  {prevalence:.3f}")
    ax_pr.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax_pr.set_title("Precision-recall curves", loc="left", fontweight="bold")
    ax_pr.legend(title="Model  PR-AUC", loc="upper right")

    max_cal = max(0.35, max(table["observed_ci_high"].max() for table in source_cal) + 0.03)
    max_cal = min(0.70, max_cal)
    ax_cal.plot([0, max_cal], [0, max_cal], color="#888888", linestyle="--", linewidth=1.0, label="Ideal")
    ax_cal.set(xlabel="Mean predicted probability", ylabel="Observed event proportion", xlim=(0, max_cal), ylim=(0, max_cal))
    ax_cal.set_title("Calibration by prediction sextile", loc="left", fontweight="bold")
    ax_cal.legend(loc="upper left", ncol=2)

    thresholds = np.linspace(0.01, 0.30, 100)
    treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    ax_dca.plot(thresholds, treat_all, color="#555555", linestyle="--", linewidth=1.2, label="Review all")
    ax_dca.axhline(0, color="#888888", linestyle=":", linewidth=1.0, label="Review none")
    ax_dca.set(xlabel="Risk threshold", ylabel="Net benefit", xlim=(0.01, 0.30), ylim=(-0.03, 0.15))
    ax_dca.set_title("Decision-curve analysis", loc="left", fontweight="bold")
    ax_dca.legend(loc="upper right", ncol=2)

    for label, axis in zip("ABCD", axes.ravel()):
        panel_label(axis, label)
        clean_axis(axis)
    save_figure(figure, outdir / "Figure3_Tier3ModelEvaluation_final")
    plt.close(figure)

    sourcedir.mkdir(parents=True, exist_ok=True)
    pd.concat(source_roc, ignore_index=True).to_csv(sourcedir / "Figure3A_ROC_source.csv", index=False)
    pd.concat(source_pr, ignore_index=True).to_csv(sourcedir / "Figure3B_PR_source.csv", index=False)
    pd.concat(source_cal, ignore_index=True).to_csv(sourcedir / "Figure3C_calibration_source.csv", index=False)
    pd.concat(source_dca, ignore_index=True).to_csv(sourcedir / "Figure3D_DCA_source.csv", index=False)


def figure4(predictions: pd.DataFrame, outdir: Path, sourcedir: Path) -> None:
    data = predictions[(predictions["Tier"] == "Tier3") & (predictions["Model"] == "CatBoost")].copy()
    threshold = float(data["locked_threshold"].iloc[0])
    low_cut = threshold / 2
    data["Risk_group"] = pd.cut(
        data["pred_prob"],
        bins=[-np.inf, low_cut, threshold, np.inf],
        labels=RISK_ORDER,
        right=False,
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), constrained_layout=True)
    ax_dist, ax_rate, ax_capture = axes

    bins = np.linspace(0, max(0.45, data["pred_prob"].quantile(0.995)), 31)
    for outcome, label, color, style in [
        (0, "No liver metastasis", "#0072B2", "-"),
        (1, "Liver metastasis", "#D55E00", "--"),
    ]:
        values = data.loc[data["y_true"] == outcome, "pred_prob"]
        ax_dist.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=color,
            linestyle=style,
            label=label,
        )
    ax_dist.axvline(low_cut, color="#777777", linestyle=":", linewidth=1.2)
    ax_dist.axvline(threshold, color="#333333", linestyle="--", linewidth=1.2)
    ax_dist.text(
        low_cut,
        0.97,
        f"{low_cut:.3f}",
        transform=ax_dist.get_xaxis_transform(),
        rotation=90,
        va="top",
        ha="right",
        fontsize=7,
        color="#666666",
    )
    ax_dist.text(
        threshold,
        0.97,
        f"{threshold:.3f}",
        transform=ax_dist.get_xaxis_transform(),
        rotation=90,
        va="top",
        ha="left",
        fontsize=7,
        color="#333333",
    )
    ax_dist.set(xlabel="Locked predicted probability", ylabel="Density")
    ax_dist.set_title("Risk distribution by outcome", loc="left", fontweight="bold")
    ax_dist.legend(loc="upper right")

    summary_rows = []
    for risk in RISK_ORDER:
        subset = data[data["Risk_group"] == risk]
        events = int(subset["y_true"].sum())
        low, high = wilson_interval(events, len(subset))
        summary_rows.append(
            {
                "Risk_group": risk,
                "N": len(subset),
                "Events": events,
                "Observed_event_rate": events / len(subset),
                "Event_rate_CI_low": low,
                "Event_rate_CI_high": high,
                "Cohort_share": len(subset) / len(data),
                "Event_capture": events / data["y_true"].sum(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    x = np.arange(len(RISK_ORDER))
    rates = summary["Observed_event_rate"].to_numpy()
    yerr = np.vstack(
        [rates - summary["Event_rate_CI_low"].to_numpy(), summary["Event_rate_CI_high"].to_numpy() - rates]
    )
    ax_rate.bar(x, rates, color=[RISK_COLORS[risk] for risk in RISK_ORDER], width=0.62)
    ax_rate.errorbar(x, rates, yerr=yerr, fmt="none", ecolor=TEXT, capsize=3, linewidth=1.0)
    for position, row in enumerate(summary.itertuples(index=False)):
        ax_rate.text(position, row.Observed_event_rate + 0.025, f"{100 * row.Observed_event_rate:.1f}%\n{row.Events}/{row.N}", ha="center", va="bottom", fontsize=8)
    ax_rate.set_xticks(x, RISK_ORDER)
    ax_rate.set(ylabel="Observed event rate", ylim=(0, max(0.32, summary["Event_rate_CI_high"].max() + 0.06)))
    ax_rate.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_rate.set_title("Observed enrichment by risk group", loc="left", fontweight="bold")

    width = 0.32
    cohort_share = summary["Cohort_share"].to_numpy()
    capture = summary["Event_capture"].to_numpy()
    ax_capture.bar(x - width / 2, cohort_share, width, color="#56B4E9", label="Cohort share")
    ax_capture.bar(x + width / 2, capture, width, color="#D55E00", label="Event capture")
    for positions, values in [(x - width / 2, cohort_share), (x + width / 2, capture)]:
        for position, value in zip(positions, values):
            ax_capture.text(position, value + 0.025, f"{100 * value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax_capture.set_xticks(x, RISK_ORDER)
    ax_capture.set(ylabel="Proportion", ylim=(0, 1.0))
    ax_capture.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax_capture.set_title("Cohort share and event capture", loc="left", fontweight="bold")
    ax_capture.legend(loc="upper left")

    for label, axis in zip("ABC", axes):
        panel_label(axis, label)
        clean_axis(axis)
    save_figure(figure, outdir / "Figure4_RiskStratification_final")
    plt.close(figure)

    sourcedir.mkdir(parents=True, exist_ok=True)
    data[["pred_prob", "y_true", "Risk_group"]].to_csv(
        sourcedir / "Figure4A_distribution_source.csv", index=False
    )
    summary.to_csv(sourcedir / "Figure4BC_risk_group_source.csv", index=False)


def add_binned_median(axis: plt.Axes, x: np.ndarray, y: np.ndarray, bins: int = 12) -> pd.DataFrame:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    frame["bin"] = pd.qcut(frame["x"], q=bins, duplicates="drop")
    medians = frame.groupby("bin", observed=True).agg(x=("x", "median"), y=("y", "median")).reset_index(drop=True)
    axis.plot(medians["x"], medians["y"], color="#D55E00", linewidth=2.0)
    return medians


def shap_dependence(shap_path: Path, outdir: Path, sourcedir: Path, seed: int) -> None:
    long = pd.read_csv(shap_path)
    long["row"] = long.groupby("feature").cumcount()
    values = long.pivot(index="row", columns="feature", values="feature_value")
    shap_values = long.pivot(index="row", columns="feature", values="shap_value")
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.4), constrained_layout=True)
    rng = np.random.default_rng(seed)
    source_rows = []

    for axis, feature, xlabel in [
        (axes[0, 0], "Age_Years", "Age, years"),
        (axes[0, 1], "Tumor_Size_Model", "Tumor size, mm"),
    ]:
        mask = values[feature].notna()
        if feature == "Tumor_Size_Model" and "Missing_Size_Indicator" in values:
            mask &= pd.to_numeric(values["Missing_Size_Indicator"], errors="coerce").eq(0)
        x = pd.to_numeric(values.loc[mask, feature], errors="coerce").to_numpy(dtype=float)
        y = shap_values.loc[mask, feature].to_numpy(dtype=float)
        axis.scatter(x, y, s=9, alpha=0.28, color="#0072B2", edgecolors="none")
        medians = add_binned_median(axis, x, y)
        axis.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
        axis.set(xlabel=xlabel, ylabel="SHAP value")
        axis.set_title(f"{xlabel} dependence", loc="left", fontweight="bold")
        source_rows.append(pd.DataFrame({"Panel": feature, "Feature_value": x, "SHAP_value": y, "Series": "Case"}))
        source_rows.append(pd.DataFrame({"Panel": feature, "Feature_value": medians["x"], "SHAP_value": medians["y"], "Series": "Binned median"}))

    categorical = [
        (axes[0, 2], "Unified_T", ["T1a", "T1b", "T1NOS", "T2"], "T subgroup"),
        (axes[1, 0], "Unified_N", ["N0", "N1", "N2", "NX"], "N subgroup"),
        (axes[1, 1], "Unified_Grade", ["G1", "G2", "G3", "G4", "Unknown"], "Grade"),
        (axes[1, 2], "Missing_Size_Indicator", ["0.0", "1.0"], "Tumor size status"),
    ]
    for axis, feature, preferred, title in categorical:
        category = values[feature].astype(str)
        available = [level for level in preferred if level in set(category)]
        if feature == "Missing_Size_Indicator":
            category = category.replace({"0.0": "Observed", "1.0": "Missing", "0": "Observed", "1": "Missing"})
            available = [level for level in ["Observed", "Missing"] if level in set(category)]
        groups = [shap_values.loc[category.eq(level), feature].dropna().to_numpy() for level in available]
        display_levels = ["T1-NOS" if level == "T1NOS" else level for level in available]
        box = axis.boxplot(groups, tick_labels=display_levels, patch_artist=True, widths=0.55, showfliers=False)
        for patch in box["boxes"]:
            patch.set(facecolor="#56B4E9", alpha=0.42, edgecolor="#0072B2")
        for median in box["medians"]:
            median.set(color="#D55E00", linewidth=1.6)
        for position, group in enumerate(groups, start=1):
            if len(group) > 350:
                group = rng.choice(group, size=350, replace=False)
            jitter = rng.uniform(-0.16, 0.16, size=len(group))
            axis.scatter(np.full(len(group), position) + jitter, group, s=7, alpha=0.18, color="#0072B2", edgecolors="none")
            source_rows.append(pd.DataFrame({"Panel": feature, "Feature_value": display_levels[position - 1], "SHAP_value": group, "Series": "Case sample"}))
        axis.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
        axis.set(xlabel=title, ylabel="SHAP value")
        axis.set_title(f"{title} dependence", loc="left", fontweight="bold")
        axis.tick_params(axis="x", rotation=25 if len(available) > 4 else 0)

    for label, axis in zip("ABCDEF", axes.ravel()):
        panel_label(axis, label)
        clean_axis(axis)
    save_figure(figure, outdir / "FigureS4_SHAPDependence_final")
    plt.close(figure)
    sourcedir.mkdir(parents=True, exist_ok=True)
    pd.concat(source_rows, ignore_index=True).to_csv(
        sourcedir / "FigureS4_SHAP_dependence_source.csv", index=False
    )


def main() -> None:
    args = parse_args()
    configure_style()
    predictions = pd.read_csv(args.predictions)
    figure3(predictions, args.figures_dir, args.source_dir)
    figure4(predictions, args.figures_dir, args.source_dir)
    shap_dependence(args.shap, args.figures_dir, args.source_dir, args.seed)
    print(f"Saved final figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
