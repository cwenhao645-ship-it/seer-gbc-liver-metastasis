"""Summarize locked-model performance across demographic subgroups.

This reporting analysis does not refit, recalibrate, or retune the model. It
uses the original Tier 3 CatBoost temporal-test predictions and reports
descriptive subgroup performance with outcome-stratified bootstrap intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


METRICS = (
    "ROC_AUC",
    "PR_AUC",
    "Brier",
    "Sensitivity",
    "Specificity",
    "PPV",
    "NPV",
)

GROUP_ORDER = {
    "Age group": ("<60 years", "60-74 years", ">=75 years", "Unknown"),
    "Sex": ("Female", "Male", "Unknown"),
    "Race/ethnicity": (
        "Non-Hispanic White",
        "Hispanic",
        "Non-Hispanic Black",
        "Asian/Pacific Islander",
        "Other/Unknown",
    ),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=root / "00_original_snapshot" / "test_set_predictions.csv",
    )
    parser.add_argument(
        "--model-ready",
        type=Path,
        default=root / "00_original_snapshot" / "01_model_ready_Tier3_fixed.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "03_analysis_outputs")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def calibration_parameters(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    if np.unique(y_true).size < 2 or int(y_true.sum()) < 20:
        return float("nan"), float("nan")
    design = sm.add_constant(logit(np.clip(probability, 1e-8, 1 - 1e-8)))
    try:
        fitted = sm.GLM(y_true, design, family=sm.families.Binomial()).fit()
    except Exception:
        return float("nan"), float("nan")
    return float(fitted.params[0]), float(fitted.params[1])


def calculate_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    include_calibration: bool = True,
) -> dict[str, float]:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    if include_calibration:
        intercept, slope = calibration_parameters(y_true, probability)
    else:
        intercept, slope = float("nan"), float("nan")
    return {
        "ROC_AUC": float(roc_auc_score(y_true, probability)),
        "PR_AUC": float(average_precision_score(y_true, probability)),
        "Brier": float(brier_score_loss(y_true, probability)),
        "Sensitivity": safe_ratio(tp, tp + fn),
        "Specificity": safe_ratio(tn, tn + fp),
        "PPV": safe_ratio(tp, tp + fp),
        "NPV": safe_ratio(tn, tn + fn),
        "Calibration_intercept": intercept,
        "Calibration_slope": slope,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def bootstrap_group(
    frame: pd.DataFrame,
    threshold: float,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    y_true = frame["y_true"].to_numpy(dtype=int)
    probability = frame["pred_prob"].to_numpy(dtype=float)
    event_indices = np.flatnonzero(y_true == 1)
    nonevent_indices = np.flatnonzero(y_true == 0)
    if not len(event_indices) or not len(nonevent_indices):
        raise ValueError("Each subgroup must contain both events and non-events")
    rows = []
    for replicate in range(1, replicates + 1):
        sample = np.concatenate(
            (
                rng.choice(event_indices, len(event_indices), replace=True),
                rng.choice(nonevent_indices, len(nonevent_indices), replace=True),
            )
        )
        metrics = calculate_metrics(
            y_true[sample], probability[sample], threshold, include_calibration=False
        )
        rows.append({"Replicate": replicate, **{name: metrics[name] for name in METRICS}})
    return pd.DataFrame(rows)


def load_temporal_predictions(args: argparse.Namespace) -> tuple[pd.DataFrame, float]:
    predictions = pd.read_csv(args.predictions)
    predictions = predictions[
        predictions["Tier"].eq("Tier3") & predictions["Model"].eq("CatBoost")
    ][["Patient_ID", "y_true", "pred_prob", "locked_threshold"]]
    if predictions["Patient_ID"].duplicated().any():
        raise AssertionError("Tier 3 CatBoost predictions contain duplicate Patient_ID values")

    model_ready = pd.read_csv(args.model_ready)
    model_ready = model_ready[
        model_ready["Temporal_Split"].str.startswith("Temporal Test")
    ][["Patient_ID", "Age_Group", "Sex", "Race_Group"]]
    merged = predictions.merge(model_ready, on="Patient_ID", validate="one_to_one")
    if len(merged) != 1934 or int(merged["y_true"].sum()) != 277:
        raise AssertionError(
            f"Expected 1,934 temporal-test records and 277 events; found "
            f"{len(merged):,} and {int(merged['y_true'].sum()):,}"
        )
    thresholds = merged["locked_threshold"].drop_duplicates().to_numpy(dtype=float)
    if len(thresholds) != 1:
        raise AssertionError("Expected one locked CatBoost threshold")
    return merged, float(thresholds[0])


def main() -> None:
    args = parse_args()
    if args.bootstrap < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    frame, threshold = load_temporal_predictions(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variable_map = {
        "Age group": "Age_Group",
        "Sex": "Sex",
        "Race/ethnicity": "Race_Group",
    }
    summary_rows = []
    replicate_rows = []
    sequence = np.random.SeedSequence(args.seed)
    group_count = sum(
        frame[column].fillna("Unknown").nunique() for column in variable_map.values()
    )
    child_seeds = iter(sequence.spawn(group_count))

    for domain, column in variable_map.items():
        values = frame[column].fillna("Unknown").astype(str)
        available = set(values.unique())
        ordered = [value for value in GROUP_ORDER[domain] if value in available]
        ordered.extend(sorted(available.difference(ordered)))
        for group in ordered:
            subset = frame.loc[values.eq(group)].copy()
            y_true = subset["y_true"].to_numpy(dtype=int)
            probability = subset["pred_prob"].to_numpy(dtype=float)
            point = calculate_metrics(y_true, probability, threshold)
            boot = bootstrap_group(
                subset,
                threshold,
                args.bootstrap,
                np.random.default_rng(next(child_seeds)),
            )
            row = {
                "Domain": domain,
                "Subgroup": group,
                "N": len(subset),
                "Events": int(y_true.sum()),
                "Event_prevalence": float(y_true.mean()),
                **point,
                "Interpretation_flag": (
                    "Sparse subgroup; interpret descriptively"
                    if int(y_true.sum()) < 20
                    else "Exploratory; not a fairness or equivalence test"
                ),
            }
            for metric in METRICS:
                row[f"{metric}_CI_low"] = float(boot[metric].quantile(0.025))
                row[f"{metric}_CI_high"] = float(boot[metric].quantile(0.975))
            summary_rows.append(row)
            boot.insert(0, "Subgroup", group)
            boot.insert(0, "Domain", domain)
            replicate_rows.append(boot)

    summary = pd.DataFrame(summary_rows)
    replicates = pd.concat(replicate_rows, ignore_index=True)
    summary_path = args.output_dir / "demographic_subgroup_performance.csv"
    replicates_path = args.output_dir / "demographic_subgroup_bootstrap_replicates.csv"
    summary.to_csv(summary_path, index=False)
    replicates.to_csv(replicates_path, index=False)

    run_summary = {
        "analysis": "Locked temporal-test demographic subgroup performance",
        "records": len(frame),
        "events": int(frame["y_true"].sum()),
        "locked_threshold": threshold,
        "bootstrap_replicates_per_subgroup": args.bootstrap,
        "seed": args.seed,
        "subgroups": len(summary),
        "model_refit": False,
        "recalibration": False,
        "threshold_retuning": False,
        "interpretation": "Exploratory reporting analysis; not a fairness or equivalence test",
    }
    summary_json = args.output_dir / "demographic_subgroup_run_summary.json"
    summary_json.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(summary_path)
    print(replicates_path)
    print(summary_json)


if __name__ == "__main__":
    main()
