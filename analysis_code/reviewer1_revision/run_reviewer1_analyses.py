"""Run all additional analyses requested by Reviewer 1.

The locked model, calibration mapping, and probability threshold are never
retrained or selected in this script. New modeling is limited to the outcome-
observation propensity model used for the selection-bias sensitivity analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from revision_common import (
    SEED,
    build_full_target_cohort,
    load_original_module,
    percentile_interval,
    score_locked_catboost,
    weighted_metrics,
)


PROPENSITY_NUMERIC = ["Year_of_diagnosis", "Age_Years", "Tumor_Size_Model"]
PROPENSITY_CATEGORICAL = [
    "Coding_Era",
    "Age_Group",
    "Sex",
    "Race_Group",
    "Marital_Status",
    "Histology_Confirmed",
    "Unified_T",
    "Unified_N",
    "Unified_Grade",
    "Missing_Size_Indicator",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Raw SEER CSV exported by the author")
    parser.add_argument(
        "--locked-dir",
        type=Path,
        default=root / "00_original_snapshot",
        help="Directory containing the locked model, summary, predictions, and Tier 3 data",
    )
    parser.add_argument(
        "--original-script",
        type=Path,
        default=root / "00_original_snapshot" / "_embedded_modelopt_calibrated.py",
    )
    parser.add_argument("--outdir", type=Path, default=root / "03_analysis_outputs")
    parser.add_argument("--tables-dir", type=Path, default=root / "04_tables")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-paired", type=int, default=5000)
    parser.add_argument("--n-mnar", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def propensity_pipeline() -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot()),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, PROPENSITY_NUMERIC),
            ("categorical", categorical, PROPENSITY_CATEGORICAL),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def fit_propensity(data: pd.DataFrame) -> tuple[Pipeline, np.ndarray]:
    model = propensity_pipeline()
    features = PROPENSITY_NUMERIC + PROPENSITY_CATEGORICAL
    model.fit(data[features], data["Outcome_Observed"].astype(int))
    probability = model.predict_proba(data[features])[:, 1]
    return model, np.clip(probability, 1e-6, 1 - 1e-6)


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / np.sum(weights**2))


def weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    mean = np.average(values, weights=weights)
    return float(np.average((values - mean) ** 2, weights=weights))


def standardized_difference(
    target_values: np.ndarray,
    observed_values: np.ndarray,
    observed_weights: np.ndarray,
) -> float:
    target_values = np.asarray(target_values, dtype=float)
    observed_values = np.asarray(observed_values, dtype=float)
    observed_weights = np.asarray(observed_weights, dtype=float)
    target_mean = float(np.mean(target_values))
    observed_mean = float(np.average(observed_values, weights=observed_weights))
    target_var = float(np.var(target_values))
    observed_var = weighted_variance(observed_values, observed_weights)
    denominator = np.sqrt((target_var + observed_var) / 2.0)
    if denominator == 0:
        return 0.0
    return float((observed_mean - target_mean) / denominator)


def balance_table(full: pd.DataFrame, observed_weights: np.ndarray) -> pd.DataFrame:
    observed = full[full["Outcome_Observed"] == 1].copy()
    rows: list[dict[str, Any]] = []
    for column in PROPENSITY_NUMERIC:
        target = full[column].fillna(full[column].median()).to_numpy(dtype=float)
        obs = observed[column].fillna(full[column].median()).to_numpy(dtype=float)
        before = standardized_difference(target, obs, np.ones(len(obs)))
        after = standardized_difference(target, obs, observed_weights)
        rows.append(
            {
                "Covariate": column,
                "Level": "Continuous",
                "Abs_SMD_before": abs(before),
                "Abs_SMD_after_IPW": abs(after),
            }
        )
    for column in PROPENSITY_CATEGORICAL:
        target_series = full[column].fillna("Missing").astype(str)
        observed_series = observed[column].fillna("Missing").astype(str)
        for level in sorted(target_series.unique()):
            target = target_series.eq(level).to_numpy(dtype=float)
            obs = observed_series.eq(level).to_numpy(dtype=float)
            before = standardized_difference(target, obs, np.ones(len(obs)))
            after = standardized_difference(target, obs, observed_weights)
            rows.append(
                {
                    "Covariate": column,
                    "Level": level,
                    "Abs_SMD_before": abs(before),
                    "Abs_SMD_after_IPW": abs(after),
                }
            )
    return pd.DataFrame(rows).sort_values("Abs_SMD_before", ascending=False).reset_index(drop=True)


def make_stabilized_weights(full: pd.DataFrame, propensity: np.ndarray) -> np.ndarray:
    observed_rate = float(full["Outcome_Observed"].mean())
    return observed_rate / propensity[full["Outcome_Observed"].to_numpy(dtype=bool)]


def selection_summary(
    full: pd.DataFrame,
    threshold: float,
    propensity: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed_mask = full["Outcome_Observed"].to_numpy(dtype=bool)
    observed = full.loc[observed_mask].copy()
    stabilized = make_stabilized_weights(full, propensity)
    lower, upper = np.quantile(stabilized, [0.01, 0.99])
    trimmed = np.clip(stabilized, lower, upper)

    test_mask = observed["Temporal_Split"].str.startswith("Temporal Test").to_numpy()
    test = observed.loc[test_mask].copy()
    test_weight = stabilized[test_mask]
    test_weight_trimmed = trimmed[test_mask]

    metric_rows = []
    for label, weights in [
        ("Complete-case temporal test", np.ones(len(test))),
        ("IPW temporal test under conditional MAR", test_weight),
        ("IPW temporal test with 1st-99th percentile weight trimming", test_weight_trimmed),
    ]:
        row = weighted_metrics(
            test["Target_Liver_Mets"], test["Locked_Probability"], threshold, weights
        )
        row["Analysis"] = label
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)

    full_test = full[full["Temporal_Split"].str.startswith("Temporal Test")].copy()
    direct = {
        "Analysis": "Directly scored full T1-T2 temporal target cohort",
        "N": int(len(full_test)),
        "Outcome_observed_n": int(full_test["Outcome_Observed"].sum()),
        "Outcome_unknown_n": int((1 - full_test["Outcome_Observed"]).sum()),
        "High_risk_n": int(full_test["Risk_Group"].eq("High").sum()),
        "High_risk_share": float(full_test["Risk_Group"].eq("High").mean()),
        "Median_probability": float(full_test["Locked_Probability"].median()),
    }
    direct_table = pd.DataFrame([direct])

    diagnostics = pd.DataFrame(
        [
            {
                "Model": "L2-penalized logistic outcome-observation model",
                "Full_target_N": len(full),
                "Observed_outcome_N": int(full["Outcome_Observed"].sum()),
                "Unknown_outcome_N": int((1 - full["Outcome_Observed"]).sum()),
                "Observed_outcome_rate": float(full["Outcome_Observed"].mean()),
                "Observation_model_ROC_AUC": float(
                    roc_auc_score(full["Outcome_Observed"], propensity)
                ),
                "Propensity_min": float(np.min(propensity)),
                "Propensity_median": float(np.median(propensity)),
                "Propensity_p01": float(np.quantile(propensity, 0.01)),
                "Propensity_p99": float(np.quantile(propensity, 0.99)),
                "Propensity_max": float(np.max(propensity)),
                "Stabilized_weight_min": float(np.min(stabilized)),
                "Stabilized_weight_median": float(np.median(stabilized)),
                "Stabilized_weight_p95": float(np.quantile(stabilized, 0.95)),
                "Stabilized_weight_p99": float(np.quantile(stabilized, 0.99)),
                "Stabilized_weight_max": float(np.max(stabilized)),
                "ESS_all_observed": effective_sample_size(stabilized),
                "ESS_temporal_test": effective_sample_size(test_weight),
                "Trim_lower": float(lower),
                "Trim_upper": float(upper),
            }
        ]
    )
    balance = balance_table(full, stabilized)

    if n_bootstrap <= 0:
        return metrics, direct_table, diagnostics, balance, pd.DataFrame()

    strata = full["Temporal_Split"].astype(str) + "|R=" + full["Outcome_Observed"].astype(str)
    stratum_indices = [np.flatnonzero(strata.to_numpy() == level) for level in sorted(strata.unique())]
    features = PROPENSITY_NUMERIC + PROPENSITY_CATEGORICAL
    preprocessor = propensity_pipeline().named_steps["preprocess"]
    transformed = preprocessor.fit_transform(full[features])
    observation = full["Outcome_Observed"].to_numpy(dtype=int)
    temporal_observed = (
        full["Temporal_Split"].str.startswith("Temporal Test").to_numpy()
        & observation.astype(bool)
    )
    temporal_target = full["Temporal_Split"].str.startswith("Temporal Test").to_numpy()
    high_risk = full["Risk_Group"].eq("High").to_numpy()
    outcome = full["Target_Liver_Mets"].fillna(0).to_numpy(dtype=int)
    probability = full["Locked_Probability"].to_numpy(dtype=float)
    seed_sequences = np.random.SeedSequence(seed).spawn(n_bootstrap)

    def one_replicate(replicate: int) -> dict[str, float]:
        rng = np.random.default_rng(seed_sequences[replicate])
        counts = np.zeros(len(full), dtype=float)
        for index in stratum_indices:
            counts[index] = rng.multinomial(len(index), np.full(len(index), 1 / len(index)))
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
        )
        model.fit(transformed, observation, sample_weight=counts)
        e = np.clip(model.predict_proba(transformed)[:, 1], 1e-6, 1 - 1e-6)
        weights = float(full["Outcome_Observed"].mean()) / e
        evaluation_weights = counts[temporal_observed] * weights[temporal_observed]
        met = weighted_metrics(
            outcome[temporal_observed],
            probability[temporal_observed],
            threshold,
            evaluation_weights,
        )
        direct_denominator = float(counts[temporal_target].sum())
        return {
            "Replicate": replicate + 1,
            **{key: float(value) for key, value in met.items() if isinstance(value, (int, float))},
            "Direct_full_target_high_risk_share": float(
                np.sum(counts[temporal_target] * high_risk[temporal_target]) / direct_denominator
            ),
        }

    jobs = max(1, min(8, os.cpu_count() or 1))
    bootstrap_rows = Parallel(n_jobs=jobs, prefer="threads")(
        delayed(one_replicate)(replicate) for replicate in range(n_bootstrap)
    )
    bootstrap = pd.DataFrame(bootstrap_rows)
    ipw_index = metrics["Analysis"].eq("IPW temporal test under conditional MAR")
    for column in [
        "Event_rate",
        "High_risk_share",
        "High_risk_event_rate",
        "Event_capture",
        "ROC_AUC",
        "PR_AUC",
        "Brier",
        "F1",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
    ]:
        low_ci, high_ci = percentile_interval(bootstrap[column])
        metrics.loc[ipw_index, f"{column}_CI_low"] = low_ci
        metrics.loc[ipw_index, f"{column}_CI_high"] = high_ci
    direct_low, direct_high = percentile_interval(
        bootstrap["Direct_full_target_high_risk_share"]
    )
    direct_table["High_risk_share_CI_low"] = direct_low
    direct_table["High_risk_share_CI_high"] = direct_high
    return metrics, direct_table, diagnostics, balance, bootstrap


def outcome_status_table(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period_label, period_data in [
        ("Overall 2010-2022", full),
        ("Development 2010-2018", full[full["Temporal_Split"].str.startswith("Development")]),
        ("Temporal test 2019-2022", full[full["Temporal_Split"].str.startswith("Temporal Test")]),
    ]:
        for observed, status in [(1, "Known outcome"), (0, "Unknown outcome")]:
            group = period_data[period_data["Outcome_Observed"] == observed]
            rows.append(
                {
                    "Period": period_label,
                    "Outcome_status": status,
                    "N": len(group),
                    "Median_predicted_probability": group["Locked_Probability"].median(),
                    "Probability_Q1": group["Locked_Probability"].quantile(0.25),
                    "Probability_Q3": group["Locked_Probability"].quantile(0.75),
                    "Mean_predicted_probability": group["Locked_Probability"].mean(),
                    "Low_risk_n": int(group["Risk_Group"].eq("Low").sum()),
                    "Intermediate_risk_n": int(group["Risk_Group"].eq("Intermediate").sum()),
                    "High_risk_n": int(group["Risk_Group"].eq("High").sum()),
                    "High_risk_share": group["Risk_Group"].eq("High").mean(),
                }
            )
    return pd.DataFrame(rows)


def risk_group_distribution(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, data in [
        ("Development 2010-2018", full[full["Temporal_Split"].str.startswith("Development")]),
        ("Temporal test 2019-2022", full[full["Temporal_Split"].str.startswith("Temporal Test")]),
    ]:
        for status, group in [
            ("All target records", data),
            ("Known outcome", data[data["Outcome_Observed"] == 1]),
            ("Unknown outcome", data[data["Outcome_Observed"] == 0]),
        ]:
            for risk in ["Low", "Intermediate", "High"]:
                count = int(group["Risk_Group"].eq(risk).sum())
                rows.append(
                    {
                        "Period": period,
                        "Outcome_status": status,
                        "Risk_group": risk,
                        "N": len(group),
                        "Risk_group_n": count,
                        "Risk_group_share": count / len(group) if len(group) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def mnar_sensitivity(
    full: pd.DataFrame,
    threshold: float,
    n_simulations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = full[full["Temporal_Split"].str.startswith("Temporal Test")].copy()
    known = target[target["Outcome_Observed"] == 1].copy()
    unknown = target[target["Outcome_Observed"] == 0].copy()
    y_known = known["Target_Liver_Mets"].to_numpy(dtype=int)
    p_known = known["Locked_Probability"].to_numpy(dtype=float)
    p_unknown = unknown["Locked_Probability"].to_numpy(dtype=float)
    high_known = p_known >= threshold
    high_unknown = p_unknown >= threshold
    known_high_events = int(y_known[high_known].sum())
    known_total_events = int(y_known.sum())
    total_high = int(high_known.sum() + high_unknown.sum())

    rng = np.random.default_rng(seed)
    rows = []
    simulation_rows = []
    deltas = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for delta in deltas:
        q = expit(logit(np.clip(p_unknown, 1e-12, 1 - 1e-12)) + delta)
        expected_missing_events = float(q.sum())
        expected_high_missing_events = float(q[high_unknown].sum())
        row: dict[str, Any] = {
            "Scenario": f"Delta {delta:+.1f}",
            "Delta_log_odds": delta,
            "Unknown_outcome_odds_multiplier": float(np.exp(delta)),
            "Unknown_test_N": len(unknown),
            "Expected_unknown_events": expected_missing_events,
            "Expected_unknown_high_risk_events": expected_high_missing_events,
            "Full_target_high_risk_share": total_high / len(target),
            "Expected_overall_event_rate": (known_total_events + expected_missing_events)
            / len(target),
            "Expected_high_risk_event_rate": (
                known_high_events + expected_high_missing_events
            )
            / total_high,
            "Expected_event_capture": (
                known_high_events + expected_high_missing_events
            )
            / (known_total_events + expected_missing_events),
        }
        simulated_metrics: dict[str, list[float]] = {
            key: [] for key in ["ROC_AUC", "PR_AUC", "Brier", "F1", "Sensitivity", "Specificity"]
        }
        for replicate in range(n_simulations):
            y_unknown = rng.binomial(1, q)
            y = np.concatenate([y_known, y_unknown])
            p = np.concatenate([p_known, p_unknown])
            met = weighted_metrics(y, p, threshold)
            for key in simulated_metrics:
                simulated_metrics[key].append(met[key])
            simulation_rows.append(
                {
                    "Scenario": row["Scenario"],
                    "Replicate": replicate + 1,
                    **{key: met[key] for key in simulated_metrics},
                }
            )
        for key, values in simulated_metrics.items():
            row[f"{key}_median"] = float(np.median(values))
            row[f"{key}_CI_low"], row[f"{key}_CI_high"] = percentile_interval(values)
        rows.append(row)

    for label, fill in [("Extreme bound all unknown non-events", 0), ("Extreme bound all unknown events", 1)]:
        y_unknown = np.full(len(unknown), fill, dtype=int)
        y = np.concatenate([y_known, y_unknown])
        p = np.concatenate([p_known, p_unknown])
        met = weighted_metrics(y, p, threshold)
        rows.append(
            {
                "Scenario": label,
                "Delta_log_odds": np.nan,
                "Unknown_outcome_odds_multiplier": np.nan,
                "Unknown_test_N": len(unknown),
                "Expected_unknown_events": int(y_unknown.sum()),
                "Expected_unknown_high_risk_events": int(y_unknown[high_unknown].sum()),
                "Full_target_high_risk_share": total_high / len(target),
                "Expected_overall_event_rate": float(y.mean()),
                "Expected_high_risk_event_rate": float(y[np.concatenate([high_known, high_unknown])].mean()),
                "Expected_event_capture": met["Sensitivity"],
                **{f"{key}_median": met[key] for key in ["ROC_AUC", "PR_AUC", "Brier", "F1", "Sensitivity", "Specificity"]},
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(simulation_rows)


def paired_model_comparisons(
    predictions_path: Path, n_bootstrap: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(predictions_path)
    tier3 = predictions[predictions["Tier"] == "Tier3"].copy()
    tier3["Patient_ID"] = tier3["Patient_ID"].astype(str)
    probabilities = tier3.pivot(index="Patient_ID", columns="Model", values="pred_prob")
    outcomes = tier3.groupby("Patient_ID")["y_true"].first().reindex(probabilities.index).astype(int)
    thresholds = tier3.groupby("Model")["locked_threshold"].first().to_dict()
    required = ["CatBoost", "Linear Logistic", "Spline Logistic", "XGBoost"]
    if not set(required).issubset(probabilities.columns):
        raise AssertionError("Tier 3 prediction file lacks one or more required candidate models")

    y = outcomes.to_numpy(dtype=int)

    def metric_vector(model: str, index: np.ndarray) -> dict[str, float]:
        yy = y[index]
        pp = probabilities[model].to_numpy(dtype=float)[index]
        pred = pp >= float(thresholds[model])
        return {
            "ROC_AUC": float(roc_auc_score(yy, pp)),
            "PR_AUC": float(average_precision_score(yy, pp)),
            "Brier": float(brier_score_loss(yy, pp)),
            "F1": float(f1_score(yy, pred, zero_division=0)),
            "High_risk_share": float(pred.mean()),
        }

    all_index = np.arange(len(y))
    point_rows = []
    for model in required:
        point_rows.append({"Model": model, **metric_vector(model, all_index), "Locked_threshold": thresholds[model]})
    points = pd.DataFrame(point_rows)

    rng = np.random.default_rng(seed)
    event_index = np.flatnonzero(y == 1)
    nonevent_index = np.flatnonzero(y == 0)
    bootstrap_rows = []
    for replicate in range(n_bootstrap):
        index = np.concatenate(
            [
                rng.choice(event_index, size=len(event_index), replace=True),
                rng.choice(nonevent_index, size=len(nonevent_index), replace=True),
            ]
        )
        cat = metric_vector("CatBoost", index)
        for comparator in ["Linear Logistic", "Spline Logistic", "XGBoost"]:
            comp = metric_vector(comparator, index)
            bootstrap_rows.append(
                {
                    "Replicate": replicate + 1,
                    "Comparator": comparator,
                    **{f"Delta_{key}_CatBoost_minus_comparator": cat[key] - comp[key] for key in cat},
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    comparison_rows = []
    cat_point = points.set_index("Model").loc["CatBoost"]
    for comparator in ["Linear Logistic", "Spline Logistic", "XGBoost"]:
        comp_point = points.set_index("Model").loc[comparator]
        subset = bootstrap[bootstrap["Comparator"] == comparator]
        row: dict[str, Any] = {"Comparator": comparator}
        for key in ["ROC_AUC", "PR_AUC", "Brier", "F1", "High_risk_share"]:
            column = f"Delta_{key}_CatBoost_minus_comparator"
            row[f"CatBoost_{key}"] = cat_point[key]
            row[f"Comparator_{key}"] = comp_point[key]
            row[column] = cat_point[key] - comp_point[key]
            row[f"{column}_CI_low"], row[f"{column}_CI_high"] = percentile_interval(subset[column])
        comparison_rows.append(row)
    return pd.DataFrame(comparison_rows), bootstrap


def format_count_percent(mask: pd.Series, denominator: int) -> str:
    count = int(mask.sum())
    return f"{count:,} ({100 * count / denominator:.1f}%)" if denominator else "NA"


def mechanism_table(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_size = full[full["Missing_Size_Indicator"] == 0].copy()
    missing_size = full[full["Missing_Size_Indicator"] == 1].copy()
    rows: list[dict[str, Any]] = []

    def add_binary(label: str, column: str, value: Any = 1) -> None:
        a = observed_size[column].eq(value)
        b = missing_size[column].eq(value)
        smd = standardized_difference(
            observed_size[column].eq(value).to_numpy(dtype=float),
            missing_size[column].eq(value).to_numpy(dtype=float),
            np.ones(len(missing_size)),
        )
        rows.append(
            {
                "Characteristic": label,
                "Observed_tumor_size": format_count_percent(a, len(observed_size)),
                "Missing_tumor_size": format_count_percent(b, len(missing_size)),
                "Abs_SMD": abs(smd),
            }
        )

    rows.append(
        {
            "Characteristic": "Cohort size, n",
            "Observed_tumor_size": f"{len(observed_size):,}",
            "Missing_tumor_size": f"{len(missing_size):,}",
            "Abs_SMD": np.nan,
        }
    )
    add_binary("Outcome status observed", "Outcome_Observed", 1)
    add_binary("Histology confirmed", "Histology_Confirmed", "Histology confirmed")
    for level in ["Performed", "Not performed", "Unknown", "Unavailable"]:
        if level in set(full["Primary_Surgery_Status"]):
            add_binary(f"Primary surgery status: {level}", "Primary_Surgery_Status", level)
    add_binary("T-NOS", "T_NOS", 1)
    add_binary("NX", "N_X", 1)
    add_binary("Grade unknown", "Grade_Unknown", 1)
    add_binary("Temporal test period", "Temporal_Split", "Temporal Test (2019-2022)")
    add_binary("Locked high-risk group", "Risk_Group", "High")

    probability_smd = standardized_difference(
        observed_size["Locked_Probability"].to_numpy(dtype=float),
        missing_size["Locked_Probability"].to_numpy(dtype=float),
        np.ones(len(missing_size)),
    )
    rows.append(
        {
            "Characteristic": "Locked predicted probability, median [IQR]",
            "Observed_tumor_size": (
                f"{observed_size['Locked_Probability'].median():.3f} "
                f"[{observed_size['Locked_Probability'].quantile(0.25):.3f}, "
                f"{observed_size['Locked_Probability'].quantile(0.75):.3f}]"
            ),
            "Missing_tumor_size": (
                f"{missing_size['Locked_Probability'].median():.3f} "
                f"[{missing_size['Locked_Probability'].quantile(0.25):.3f}, "
                f"{missing_size['Locked_Probability'].quantile(0.75):.3f}]"
            ),
            "Abs_SMD": abs(probability_smd),
        }
    )
    known_observed_size = observed_size[observed_size["Outcome_Observed"] == 1]
    known_missing_size = missing_size[missing_size["Outcome_Observed"] == 1]
    rows.append(
        {
            "Characteristic": "Liver metastasis among known outcomes, n (%)",
            "Observed_tumor_size": format_count_percent(
                known_observed_size["Target_Liver_Mets"].eq(1), len(known_observed_size)
            ),
            "Missing_tumor_size": format_count_percent(
                known_missing_size["Target_Liver_Mets"].eq(1), len(known_missing_size)
            ),
            "Abs_SMD": np.nan,
        }
    )

    uncertainty_rows = []
    for signal, mask in [
        ("T-NOS", full["T_NOS"].eq(1)),
        ("NX", full["N_X"].eq(1)),
        ("Grade unknown", full["Grade_Unknown"].eq(1)),
        ("Tumor size missing", full["Tumor_Size_Missing"].eq(1)),
        ("No recorded uncertainty signal", full[["T_NOS", "N_X", "Grade_Unknown", "Tumor_Size_Missing"]].sum(axis=1).eq(0)),
    ]:
        group = full[mask].copy()
        known = group[group["Outcome_Observed"] == 1]
        uncertainty_rows.append(
            {
                "Signal": signal,
                "N": len(group),
                "Outcome_observed_n": int(group["Outcome_Observed"].sum()),
                "Outcome_observed_share": float(group["Outcome_Observed"].mean()),
                "Median_locked_probability": float(group["Locked_Probability"].median()),
                "High_risk_share": float(group["Risk_Group"].eq("High").mean()),
                "Known_outcome_event_rate": float(known["Target_Liver_Mets"].mean()),
                "Histology_confirmed_share": float(group["Histology_Confirmed"].eq("Histology confirmed").mean()),
                "Surgery_performed_share": float(group["Primary_Surgery_Status"].eq("Performed").mean()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(uncertainty_rows)


def outcome_status_smd_table(full: pd.DataFrame) -> pd.DataFrame:
    known = full[full["Outcome_Observed"] == 1].copy()
    unknown = full[full["Outcome_Observed"] == 0].copy()
    rows = [
        {
            "Characteristic": "Cohort size, n",
            "Known_outcome": f"{len(known):,}",
            "Unknown_outcome": f"{len(unknown):,}",
            "Abs_SMD": np.nan,
        }
    ]
    for column in PROPENSITY_NUMERIC:
        target = known[column].fillna(full[column].median()).to_numpy(dtype=float)
        values = unknown[column].fillna(full[column].median()).to_numpy(dtype=float)
        smd = standardized_difference(target, values, np.ones(len(values)))
        rows.append(
            {
                "Characteristic": f"{column}, median [IQR]",
                "Known_outcome": f"{known[column].median():.1f} [{known[column].quantile(.25):.1f}, {known[column].quantile(.75):.1f}]",
                "Unknown_outcome": f"{unknown[column].median():.1f} [{unknown[column].quantile(.25):.1f}, {unknown[column].quantile(.75):.1f}]",
                "Abs_SMD": abs(smd),
            }
        )
    for column in PROPENSITY_CATEGORICAL + ["Primary_Surgery_Status", "Risk_Group"]:
        for level in sorted(full[column].fillna("Missing").astype(str).unique()):
            known_mask = known[column].fillna("Missing").astype(str).eq(level)
            unknown_mask = unknown[column].fillna("Missing").astype(str).eq(level)
            target = known[column].fillna("Missing").astype(str).eq(level).to_numpy(dtype=float)
            smd = standardized_difference(target, unknown_mask.to_numpy(dtype=float), np.ones(len(unknown)))
            rows.append(
                {
                    "Characteristic": f"{column}: {level}",
                    "Known_outcome": format_count_percent(known_mask, len(known)),
                    "Unknown_outcome": format_count_percent(unknown_mask, len(unknown)),
                    "Abs_SMD": abs(smd),
                }
            )
    return pd.DataFrame(rows)


def save_excel(path: Path, sheets: dict[str, pd.DataFrame], notes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
        pd.DataFrame({"Notes": notes}).to_excel(writer, sheet_name="Notes", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                width = min(60, max(11, max(len(str(cell.value or "")) for cell in column_cells) + 2))
                worksheet.column_dimensions[column_cells[0].column_letter].width = width


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    module = load_original_module(args.original_script)
    raw = pd.read_csv(args.raw, low_memory=False)
    full = build_full_target_cohort(raw, module)
    full, lock_diagnostics = score_locked_catboost(full, args.locked_dir, module)
    threshold = float(lock_diagnostics["locked_threshold"])

    _, propensity = fit_propensity(full)
    metrics, direct, propensity_diagnostics, balance, ipw_bootstrap = selection_summary(
        full, threshold, propensity, args.n_bootstrap, args.seed
    )
    outcome_status = outcome_status_table(full)
    risk_groups = risk_group_distribution(full)
    outcome_smd = outcome_status_smd_table(full)

    mnar, mnar_simulations = mnar_sensitivity(
        full, threshold, args.n_mnar, args.seed + 1
    )
    data_dir = args.locked_dir / "data" if (args.locked_dir / "data").is_dir() else args.locked_dir
    paired, paired_bootstrap = paired_model_comparisons(
        data_dir / "test_set_predictions.csv", args.n_paired, args.seed + 2
    )
    mechanism, uncertainty = mechanism_table(full)

    output_tables = {
        "selection_metrics": metrics,
        "direct_target_scoring": direct,
        "propensity_diagnostics": propensity_diagnostics,
        "covariate_balance": balance,
        "outcome_status_summary": outcome_status,
        "risk_group_distribution": risk_groups,
        "target_outcome_status_SMD": outcome_smd,
        "mnar_scenarios": mnar,
        "paired_model_comparisons": paired,
        "tumor_size_mechanism": mechanism,
        "uncertainty_signals": uncertainty,
    }
    for name, table in output_tables.items():
        table.to_csv(args.outdir / f"{name}.csv", index=False, encoding="utf-8-sig")

    ipw_bootstrap.to_csv(
        args.outdir / "ipw_bootstrap_replicates.csv", index=False, encoding="utf-8-sig"
    )
    mnar_simulations.to_csv(
        args.outdir / "mnar_simulation_replicates.csv", index=False, encoding="utf-8-sig"
    )
    paired_bootstrap.to_csv(
        args.outdir / "paired_model_bootstrap_replicates.csv", index=False, encoding="utf-8-sig"
    )

    save_excel(
        args.tables_dir / "Table_S11_SelectionBias_IPW.xlsx",
        {
            "Summary": metrics,
            "Direct scoring": direct,
            "Weight diagnostics": propensity_diagnostics,
            "Covariate balance": balance,
            "Outcome status": outcome_status,
            "Risk groups": risk_groups,
            "Target cohort SMD": outcome_smd,
        },
        [
            "The estimand is the harmonized T1-T2 target cohort before exclusion for unknown liver-metastasis status.",
            "Inverse-probability weights estimate complete-case performance under conditional missing at random given the listed registry covariates.",
            "The analysis does not establish that outcome missingness is missing at random and does not identify unrestricted MNAR mechanisms.",
            "The CatBoost model, sigmoid calibration map, and probability threshold were kept locked.",
            "Direct full-target risk-group proportions require no assumption about the unobserved outcomes.",
        ],
    )
    save_excel(
        args.tables_dir / "Table_S12_MNAR_PatternMixture.xlsx",
        {"Pattern mixture": mnar},
        [
            "Delta shifts the log odds of liver metastasis for records with unknown outcomes relative to their locked-model probability.",
            "These are sensitivity scenarios rather than identified estimates of the missing outcomes.",
            "Simulation intervals describe Bernoulli imputations under each specified delta and are not confidence intervals for an identified MNAR parameter.",
            "Extreme bounds assign all 17 unknown temporal-test outcomes to non-events or events.",
        ],
    )
    save_excel(
        args.tables_dir / "Table_S13_PairedModelComparisons.xlsx",
        {"Paired comparisons": paired},
        [
            "All differences use the same 1,934 temporal-test records and paired, outcome-stratified bootstrap resampling.",
            "Differences are CatBoost minus comparator; a negative Brier-score difference favors CatBoost.",
            "F1 and high-risk-share comparisons use each model's own development-locked threshold and are threshold dependent.",
            "The temporal test cohort was not used to retrain, retune, or reselect any model or threshold.",
        ],
    )
    save_excel(
        args.tables_dir / "Table_S14_UncertaintySignalMechanism.xlsx",
        {"Tumor size missingness": mechanism, "Uncertainty signals": uncertainty},
        [
            "These comparisons are exploratory descriptions of registry data-generating pathways and do not establish causality.",
            "Primary surgery status is derived from the SEER reason-no-cancer-directed-surgery field and occurs after diagnostic assessment.",
            "Observed event rates are calculated only among records with known liver-metastasis status.",
        ],
    )

    unknown_raw_status = (
        full.loc[full["Outcome_Observed"] == 0, "SEER Combined Mets at DX-liver (2010+)"]
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .to_dict()
    )
    run_summary = {
        "status": "PASS",
        "seed": args.seed,
        "raw_records": int(len(raw)),
        "full_T1_T2_target_records": int(len(full)),
        "known_outcome_records": int(full["Outcome_Observed"].sum()),
        "unknown_outcome_records": int((1 - full["Outcome_Observed"]).sum()),
        "development_unknown_outcomes": int(
            ((full["Outcome_Observed"] == 0) & full["Temporal_Split"].str.startswith("Development")).sum()
        ),
        "temporal_test_unknown_outcomes": int(
            ((full["Outcome_Observed"] == 0) & full["Temporal_Split"].str.startswith("Temporal Test")).sum()
        ),
        "unknown_raw_statuses": unknown_raw_status,
        "lock_verification": lock_diagnostics,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "replicates": {
            "ipw_bootstrap": args.n_bootstrap,
            "paired_model_bootstrap": args.n_paired,
            "mnar_imputations_per_delta": args.n_mnar,
        },
    }
    (args.outdir / "revision_analysis_run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
