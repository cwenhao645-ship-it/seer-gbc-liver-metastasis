"""Shared utilities for the Reviewer 1 revision analyses.

The original model-development script remains the source of truth for variable
harmonization. This module imports those functions, reconstructs the T1-T2
target cohort before outcome-status exclusion, and verifies that the known-
outcome subset reproduces the locked analysis data exactly.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


SEED = 20260721


def load_original_module(path: Path) -> ModuleType:
    """Load the locked analysis module without modifying it."""
    spec = importlib.util.spec_from_file_location("locked_original_analysis", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import locked analysis script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cohort_args() -> SimpleNamespace:
    """Return the locked cohort-construction settings used in the manuscript."""
    return SimpleNamespace(
        dev_years_start=2010,
        dev_years_end=2018,
        test_years_start=2019,
        test_years_end=2022,
        adenocarcinoma_only=False,
        fine_t2_main=False,
        treat_zero_size_missing=False,
        exclude_years="",
    )


def build_full_target_cohort(raw: pd.DataFrame, module: ModuleType) -> pd.DataFrame:
    """Build the harmonized T1-T2 cohort before excluding unknown outcomes."""
    args = cohort_args()
    df = raw.copy()
    df["Year_of_diagnosis"] = df["Year of diagnosis"].apply(module.as_year)
    df = df[
        df["Year_of_diagnosis"].between(
            args.dev_years_start, args.test_years_end, inclusive="both"
        )
    ].copy()

    df["Target_Liver_Mets"] = df["SEER Combined Mets at DX-liver (2010+)"].apply(
        module.map_liver_mets
    )
    df["Outcome_Observed"] = df["Target_Liver_Mets"].isin([0, 1]).astype(int)
    df["Unified_T"] = df.apply(
        lambda row: module.map_t_stage(row, collapse_t2=True), axis=1
    )
    df["Unified_N"] = df.apply(module.map_n_stage, axis=1)
    df["Unified_Grade"] = df.apply(module.map_grade, axis=1)
    df["Tumor_Size_Observed"] = df.apply(
        lambda row: module.map_tumor_size(row, treat_zero_missing=False), axis=1
    )
    df["Missing_Size_Indicator"] = df["Tumor_Size_Observed"].isna().astype(int)
    df = df[df["Unified_T"].notna()].copy()

    df["Age_Years"] = df["Age recode with <1 year olds and 90+"].apply(
        module.parse_age_years
    )
    df["Age_Group"] = df["Age_Years"].apply(module.age_group)
    df["Race_Group"] = df[
        "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)"
    ].apply(module.map_race_origin)
    df["Marital_Status"] = df["Marital status at diagnosis"].apply(
        module.map_marital_status
    )
    df["Histology_Confirmed"] = df["Diagnostic Confirmation"].apply(
        module.map_histology_confirmed
    )
    df["Sex"] = (
        df["Sex"]
        .fillna("Unknown")
        .astype(str)
        .replace({"Blank(s)": "Unknown", "": "Unknown"})
    )
    df["Coding_Era"] = df["Year_of_diagnosis"].astype(int).apply(module.coding_era)
    df["Temporal_Split"] = np.where(
        df["Year_of_diagnosis"].between(2010, 2018),
        "Development (2010-2018)",
        "Temporal Test (2019-2022)",
    )
    if "Patient ID" in df.columns:
        df["Patient_ID"] = df["Patient ID"].astype(str)
    else:
        df["Patient_ID"] = [f"P{i:08d}" for i in range(len(df))]

    df["T_NOS"] = df["Unified_T"].astype(str).str.contains("NOS", na=False).astype(int)
    df["N_X"] = (df["Unified_N"] == "NX").astype(int)
    df["Grade_Unknown"] = (df["Unified_Grade"] == "Unknown").astype(int)
    df["Tumor_Size_Missing"] = df["Missing_Size_Indicator"].astype(int)

    reason_col = "Reason no cancer-directed surgery"
    if reason_col in df.columns:
        reason = df[reason_col].fillna("Unknown").astype(str)
        df["Primary_Surgery_Status"] = np.select(
            [
                reason.eq("Surgery performed"),
                reason.str.startswith("Recommended but not performed", na=False)
                | reason.str.startswith("Not recommended", na=False)
                | reason.str.startswith("Not performed", na=False),
            ],
            ["Performed", "Not performed"],
            default="Unknown",
        )
    else:
        df["Primary_Surgery_Status"] = "Unavailable"

    return df.reset_index(drop=True)


def add_locked_size_imputation(
    full: pd.DataFrame, locked_tier3: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    """Apply the development-cohort tumor-size median from the locked dataset."""
    dev = locked_tier3["Temporal_Split"].astype(str).str.startswith("Development")
    median = float(locked_tier3.loc[dev, "Tumor_Size_Observed"].median(skipna=True))
    out = full.copy()
    out["Tumor_Size_Model"] = out["Tumor_Size_Observed"].fillna(median)
    return out, median


def verify_known_cohort(full: pd.DataFrame, locked_tier3: pd.DataFrame) -> dict[str, Any]:
    """Assert exact reconstruction of the locked known-outcome Tier 3 cohort."""
    reconstructed = full[full["Outcome_Observed"] == 1].copy()
    locked = locked_tier3.copy()
    reconstructed["Patient_ID"] = reconstructed["Patient_ID"].astype(str)
    locked["Patient_ID"] = locked["Patient_ID"].astype(str)
    if len(reconstructed) != len(locked):
        raise AssertionError(
            f"Known cohort size mismatch: {len(reconstructed)} versus {len(locked)}"
        )
    if set(reconstructed["Patient_ID"]) != set(locked["Patient_ID"]):
        raise AssertionError("Patient-ID sets differ between reconstructed and locked cohorts")

    compare_cols = [
        "Year_of_diagnosis",
        "Temporal_Split",
        "Coding_Era",
        "Target_Liver_Mets",
        "Age_Years",
        "Age_Group",
        "Sex",
        "Race_Group",
        "Marital_Status",
        "Histology_Confirmed",
        "Unified_T",
        "Unified_N",
        "Unified_Grade",
        "Tumor_Size_Model",
        "Missing_Size_Indicator",
        "Tumor_Size_Observed",
        "T_NOS",
        "N_X",
        "Grade_Unknown",
        "Tumor_Size_Missing",
    ]
    left = reconstructed.set_index("Patient_ID")[compare_cols].sort_index()
    right = locked.set_index("Patient_ID")[compare_cols].sort_index()
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_like=True)
    return {
        "known_cohort_n": int(len(locked)),
        "columns_verified": compare_cols,
        "status": "PASS",
    }


def recover_locked_calibration(
    raw_prob: np.ndarray, calibrated_prob: np.ndarray
) -> tuple[float, float, float]:
    """Recover the saved sigmoid calibration map and verify exact reproduction."""
    x = logit(np.clip(np.asarray(raw_prob, dtype=float), 1e-12, 1 - 1e-12))
    y = logit(np.clip(np.asarray(calibrated_prob, dtype=float), 1e-12, 1 - 1e-12))
    design = np.column_stack([np.ones(len(x)), x])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    reconstructed = expit(alpha + beta * x)
    max_error = float(np.max(np.abs(reconstructed - calibrated_prob)))
    if max_error > 1e-10:
        raise AssertionError(f"Locked calibration was not reproduced: max error={max_error}")
    return float(alpha), float(beta), max_error


def score_locked_catboost(
    full: pd.DataFrame,
    locked_dir: Path,
    module: ModuleType,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score all target-cohort records with the unchanged locked CatBoost model."""
    summary = json.loads((locked_dir / "selected_model_summary.json").read_text(encoding="utf-8"))
    feature_cols = list(summary["feature_columns"])
    threshold = float(summary["locked_threshold"])
    model = joblib.load(locked_dir / "selected_model.pkl")

    data_dir = locked_dir / "data" if (locked_dir / "data").is_dir() else locked_dir
    locked_tier3 = pd.read_csv(data_dir / "01_model_ready_Tier3_fixed.csv")
    full, size_median = add_locked_size_imputation(full, locked_tier3)
    verification = verify_known_cohort(full, locked_tier3)

    known_dev = locked_tier3[
        locked_tier3["Temporal_Split"].astype(str).str.startswith("Development")
    ]
    numeric_cols, categorical_cols = module.split_num_cat(locked_tier3, feature_cols)
    _, dev_medians, _ = module.prepare_catboost_frame(
        known_dev[feature_cols], feature_cols, numeric_cols, categorical_cols
    )
    prepared, _, _ = module.prepare_catboost_frame(
        full[feature_cols],
        feature_cols,
        numeric_cols,
        categorical_cols,
        dev_medians=dev_medians,
    )
    full = full.copy()
    full["Locked_Raw_Probability"] = model.predict_proba(prepared)[:, 1]

    saved = pd.read_csv(data_dir / "test_set_predictions.csv")
    saved = saved[(saved["Tier"] == "Tier3") & (saved["Model"] == "CatBoost")].copy()
    saved["Patient_ID"] = saved["Patient_ID"].astype(str)
    test = full[
        (full["Outcome_Observed"] == 1)
        & full["Temporal_Split"].astype(str).str.startswith("Temporal Test")
    ][["Patient_ID", "Locked_Raw_Probability"]].copy()
    test["Patient_ID"] = test["Patient_ID"].astype(str)
    aligned = test.merge(
        saved[["Patient_ID", "pred_prob"]], on="Patient_ID", how="inner", validate="one_to_one"
    )
    if len(aligned) != len(saved):
        raise AssertionError("Could not align all locked temporal-test predictions")
    alpha, beta, max_error = recover_locked_calibration(
        aligned["Locked_Raw_Probability"].to_numpy(), aligned["pred_prob"].to_numpy()
    )
    full["Locked_Probability"] = expit(
        alpha
        + beta
        * logit(np.clip(full["Locked_Raw_Probability"].to_numpy(), 1e-12, 1 - 1e-12))
    )
    low_cut = threshold / 2.0
    full["Risk_Group"] = pd.cut(
        full["Locked_Probability"],
        bins=[-np.inf, low_cut, threshold, np.inf],
        labels=["Low", "Intermediate", "High"],
        right=False,
    ).astype(str)
    diagnostics = {
        **verification,
        "locked_threshold": threshold,
        "locked_low_cut": low_cut,
        "locked_tumor_size_development_median": size_median,
        "calibration_intercept": alpha,
        "calibration_slope": beta,
        "calibration_max_absolute_reconstruction_error": max_error,
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }
    return full, diagnostics


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    return float(np.sum(values_array * weights_array) / np.sum(weights_array))


def weighted_metrics(
    y: Iterable[int], p: Iterable[float], threshold: float, weights: Iterable[float] | None = None
) -> dict[str, float]:
    """Compute probability and locked-threshold metrics with optional weights."""
    y_array = np.asarray(y, dtype=int)
    p_array = np.asarray(p, dtype=float)
    w = np.ones(len(y_array), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    pred = (p_array >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_array, pred, labels=[0, 1], sample_weight=w
    ).ravel()
    total = float(np.sum(w))
    events = float(np.sum(w * y_array))
    predicted_positive = float(np.sum(w * pred))
    result = {
        "N_unweighted": int(len(y_array)),
        "Weighted_N": total,
        "Weighted_events": events,
        "Event_rate": events / total,
        "High_risk_share": predicted_positive / total,
        "High_risk_event_rate": tp / (tp + fp) if (tp + fp) else math.nan,
        "Event_capture": tp / (tp + fn) if (tp + fn) else math.nan,
        "ROC_AUC": float(roc_auc_score(y_array, p_array, sample_weight=w)),
        "PR_AUC": float(average_precision_score(y_array, p_array, sample_weight=w)),
        "Brier": float(brier_score_loss(y_array, p_array, sample_weight=w)),
        "F1": float(f1_score(y_array, pred, sample_weight=w, zero_division=0)),
        "Sensitivity": tp / (tp + fn) if (tp + fn) else math.nan,
        "Specificity": tn / (tn + fp) if (tn + fp) else math.nan,
        "PPV": tp / (tp + fp) if (tp + fp) else math.nan,
        "NPV": tn / (tn + fn) if (tn + fn) else math.nan,
        "TP": float(tp),
        "FP": float(fp),
        "TN": float(tn),
        "FN": float(fn),
    }
    return result


def percentile_interval(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan
    low, high = np.quantile(array, [0.025, 0.975])
    return float(low), float(high)
