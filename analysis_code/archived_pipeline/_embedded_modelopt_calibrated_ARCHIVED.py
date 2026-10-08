#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rebuild_new_article_modelopt_calibrated_topjournal.py

Corrected one-click SEER T1-T2 gallbladder cancer initial liver-metastasis
risk-stratification pipeline.

Major fixes implemented compared with earlier drafts:
1) Exact marital-status recoding. Prevents "Single (never married)" being misclassified as Married.
2) Race/origin recoding based on "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)".
3) 2016-2017 SEER Combined T/N mapping included by default, avoiding silent temporal gaps.
4) Primary analysis collapses T2a/T2b/T2NOS to T2 by default to reduce coding-era drift.
5) Tumor size is split into Tumor_Size_Observed, Tumor_Size_Model, and Missing_Size_Indicator.
6) Tier 2 low-event warnings and conservative search-size reduction.
7) Table 3, Figure 4, Table S7, and selected model are all generated from the same tier-specific prediction dataframe.
8) Same-cohort uncertainty-masked ablation is separated from Tier 2 vs Tier 3 comparison.

Example:
    python rebuild_new_article_final_fixed.py ^
        --workdir . ^
        --input newgla_reextract.csv ^
        --outdir final_submission_suite_fixed ^
        --cv-folds 5 ^
        --search-iters-xgb 30 ^
        --search-iters-cat 30 ^
        --bootstrap 1000

Debug run:
    python rebuild_new_article_final_fixed.py --workdir . --fast --skip-shap
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import randint, uniform, loguniform

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.isotonic import IsotonicRegression

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    XGBClassifier = None
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except Exception:
    CatBoostClassifier = None
    HAS_CAT = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False

try:
    import statsmodels.api as sm
    HAS_SM = True
except Exception:
    sm = None
    HAS_SM = False


# -----------------------------------------------------------------------------
# Journal-style figure system
# -----------------------------------------------------------------------------

PUB = {
    "navy": "#073763",
    "blue": "#1F4E79",
    "light_blue": "#EAF2FB",
    "teal": "#0F8B8D",
    "orange": "#F26B21",
    "gray": "#6E7781",
    "light_gray": "#EEF2F5",
    "line_gray": "#B8C4D0",
    "text": "#102033",
}


def set_publication_style() -> None:
    """Global matplotlib settings for consistent journal-like output."""
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "axes.edgecolor": "#263849",
        "axes.labelcolor": PUB["text"],
        "axes.titlecolor": PUB["text"],
        "xtick.color": PUB["text"],
        "ytick.color": PUB["text"],
        "text.color": PUB["text"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#D9E1EA",
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "legend.frameon": True,
        "legend.framealpha": 0.94,
        "legend.edgecolor": "#B8C4D0",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(
        x, y, label, transform=ax.transAxes, ha="center", va="center",
        fontsize=12, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.22", facecolor=PUB["navy"], edgecolor=PUB["navy"], linewidth=0),
        clip_on=False, zorder=20,
    )


def add_panel_box(ax: plt.Axes) -> None:
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch(
        (-0.08, -0.16), 1.16, 1.30, transform=ax.transAxes, clip_on=False,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor="none", edgecolor="#B8C4D0", linewidth=0.8, zorder=-10,
    )
    ax.add_patch(box)



def save_pub_figure(fig: plt.Figure, outpath: Path) -> None:
    """Save each figure as PDF, PNG, TIFF, and editable SVG.

    SVG is intentionally exported for manual post-processing in Illustrator,
    Inkscape, Affinity Designer, or recent PowerPoint versions. The PDF is also
    vector-editable in most journal workflows. PNG/TIFF are raster deliverables.
    """
    outpath = Path(outpath)
    ensure_dir(outpath.parent)
    stem = outpath.with_suffix("")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=600)
    fig.savefig(stem.with_suffix(".tiff"), bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    # Keep an explicit editable copy folder so the editable assets are easy to find.
    editable_dir = ensure_dir(outpath.parent / "editable_svg")
    fig.savefig(editable_dir / (stem.name + ".svg"), bbox_inches="tight")

def fmt_n(x: Any) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return str(x)


def fmt3(x: Any) -> str:
    try:
        return f"{float(x):.3f}"
    except Exception:
        return "NA"


def get_model_palette(models: Sequence[str]) -> Dict[str, str]:
    base = {
        "Linear Logistic": PUB["navy"],
        "Spline Logistic": PUB["teal"],
        "XGBoost": PUB["orange"],
        "CatBoost": "#7B4AB8",
    }
    fallback = [PUB["navy"], PUB["teal"], PUB["orange"], "#7B4AB8", "#4C78A8", "#72B7B2"]
    return {m: base.get(m, fallback[i % len(fallback)]) for i, m in enumerate(models)}


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class ModelResult:
    tier: str
    model: str
    estimator: Any
    threshold: float
    threshold_rule: str
    oof_pred: np.ndarray
    test_pred: np.ndarray
    y_dev: np.ndarray
    y_test: np.ndarray
    x_dev: pd.DataFrame
    x_test: pd.DataFrame
    feature_cols: List[str]
    numeric_cols: List[str]
    categorical_cols: List[str]
    selected_params: Dict[str, Any]
    cv_score: Optional[float]
    warnings: List[str]


META_COLS = [
    "Patient_ID",
    "Year_of_diagnosis",
    "Temporal_Split",
    "Coding_Era",
]
TARGET_COL = "Target_Liver_Mets"

TIER1_FEATURES = [
    "Age_Years",
    "Age_Group",
    "Sex",
    "Race_Group",
    "Marital_Status",
]

TIER3_FEATURES = [
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
]

TIER2_FEATURES = [
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
]

UNCERTAINTY_FEATURES = [
    "T_NOS",
    "N_X",
    "Grade_Unknown",
    "Tumor_Size_Missing",
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def clean_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_space(x: Any) -> str:
    return re.sub(r"\s+", " ", clean_text(x))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_year(s: Any) -> Optional[int]:
    try:
        if pd.isna(s):
            return None
        return int(float(s))
    except Exception:
        m = re.search(r"(19|20)\d{2}", str(s))
        return int(m.group(0)) if m else None


def safe_auc(y: Sequence[int], p: Sequence[float]) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def safe_ap(y: Sequence[int], p: Sequence[float]) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))


def logit_clip(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def infer_input_file(workdir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = workdir / p
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        return p

    candidates = [
        "newgla_reextract.csv",
        "newgla_reextract(3).csv",
        "newgla_reextract(2).csv",
        "newgla_reextract(1).csv",
    ]
    for name in candidates:
        p = workdir / name
        if p.exists():
            return p
    csvs = sorted(workdir.glob("*.csv"), key=lambda x: x.stat().st_size, reverse=True)
    if not csvs:
        raise FileNotFoundError("No CSV input found. Provide --input.")
    return csvs[0]


def strip_prefix_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in params.items():
        out[k.replace("clf__", "")] = v
    return out


# -----------------------------------------------------------------------------
# SEER harmonization
# -----------------------------------------------------------------------------

def map_liver_mets(x: Any) -> Optional[int]:
    s = normalize_space(x).lower()
    if s == "yes":
        return 1
    if s == "no":
        return 0
    return None


def parse_age_years(x: Any) -> float:
    s = normalize_space(x)
    if not s or s.lower() == "unknown":
        return np.nan
    if "90+" in s:
        return 90.0
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2.0
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def age_group(age: float) -> str:
    if pd.isna(age):
        return "Unknown"
    if age < 60:
        return "<60 years"
    if age < 75:
        return "60-74 years"
    return ">=75 years"


def map_race_origin(x: Any) -> str:
    s = normalize_space(x)
    if s == "Non-Hispanic White":
        return "Non-Hispanic White"
    if s == "Non-Hispanic Black":
        return "Non-Hispanic Black"
    if s == "Hispanic (All Races)":
        return "Hispanic"
    if s == "Non-Hispanic Asian or Pacific Islander":
        return "Asian/Pacific Islander"
    if s in ["Non-Hispanic American Indian/Alaska Native", "Non-Hispanic Unknown Race", "", "Unknown"]:
        return "Other/Unknown"
    return "Other/Unknown"


def map_marital_status(x: Any) -> str:
    s = normalize_space(x)
    if s == "Married (including common law)":
        return "Married"
    if s in ["Single (never married)", "Unmarried or Domestic Partner"]:
        return "Unmarried/Single"
    if s in ["Divorced", "Separated", "Widowed"]:
        return "Divorced/Separated/Widowed"
    if s in ["Unknown", "", "Blank(s)"]:
        return "Unknown"
    return "Unknown"


def map_histology_confirmed(x: Any) -> str:
    s = normalize_space(x)
    if s in [
        "Positive histology",
        "Pos hist AND immunophenotyping AND/OR pos genetic studies",
    ]:
        return "Histology confirmed"
    if s in [
        "Positive exfoliative cytology, no positive histology",
        "Positive microscopic confirm, method not specified",
    ]:
        return "Other microscopic confirmation"
    if s in [
        "Radiography without microscopic confirm",
        "Clinical diagnosis only",
        "Direct visualization without microscopic confirmation",
        "Positive laboratory test/marker study",
        "Unknown",
        "",
    ]:
        return "Non-microscopic/clinical/unknown"
    return "Non-microscopic/clinical/unknown"


def choose_by_year(row: pd.Series, col_2010_2015: str, col_2016_2017: str, col_2018: str) -> Any:
    year = as_year(row.get("Year of diagnosis"))
    if year is None:
        return None
    if year <= 2015:
        return row.get(col_2010_2015)
    if year in [2016, 2017]:
        return row.get(col_2016_2017)
    return row.get(col_2018)


def map_t_stage(row: pd.Series, collapse_t2: bool = True) -> Optional[str]:
    year = as_year(row.get("Year of diagnosis"))
    if year is None:
        return None

    if year <= 2015:
        raw = normalize_space(row.get("Derived AJCC T, 7th ed (2010-2015)"))
        mapping = {
            "T1a": "T1a",
            "T1b": "T1b",
            "T1NOS": "T1NOS",
            "T1": "T1NOS",
            "T2": "T2" if collapse_t2 else "T2NOS",
        }
        return mapping.get(raw)

    if year in [2016, 2017]:
        raw = normalize_space(row.get("Derived SEER Combined T (2016-2017)"))
        raw_l = raw.lower()
        mapping = {
            "c1a": "T1a",
            "p1a": "T1a",
            "c1b": "T1b",
            "p1b": "T1b",
            "c1": "T1NOS",
            "p1": "T1NOS",
            "c2": "T2" if collapse_t2 else "T2NOS",
            "p2": "T2" if collapse_t2 else "T2NOS",
        }
        return mapping.get(raw_l)

    raw = normalize_space(row.get("Derived EOD 2018 T Recode (2018+)"))
    mapping = {
        "T1a": "T1a",
        "T1b": "T1b",
        "T1": "T1NOS",
        "T2": "T2" if collapse_t2 else "T2NOS",
        "T2a": "T2" if collapse_t2 else "T2a",
        "T2b": "T2" if collapse_t2 else "T2b",
    }
    return mapping.get(raw)


def map_n_stage(row: pd.Series) -> Optional[str]:
    year = as_year(row.get("Year of diagnosis"))
    if year is None:
        return None

    if year <= 2015:
        raw = normalize_space(row.get("Derived AJCC N, 7th ed (2010-2015)"))
        if raw in ["N0", "N1", "N2", "NX"]:
            return raw
        return None

    if year in [2016, 2017]:
        raw = normalize_space(row.get("Derived SEER Combined N (2016-2017)")).lower()
        mapping = {
            "c0": "N0", "p0": "N0",
            "c1": "N1", "p1": "N1",
            "c2": "N2", "p2": "N2",
            "cx": "NX", "px": "NX",
        }
        return mapping.get(raw)

    raw = normalize_space(row.get("Derived EOD 2018 N Recode (2018+)"))
    if raw in ["N0", "N1", "N2", "NX"]:
        return raw
    return None


def map_grade(row: pd.Series) -> str:
    year = as_year(row.get("Year of diagnosis"))
    if year is None:
        return "Unknown"

    if year <= 2017:
        raw = normalize_space(row.get("Grade Recode (thru 2017)"))
        if raw.startswith("Well differentiated"):
            return "G1"
        if raw.startswith("Moderately differentiated"):
            return "G2"
        if raw.startswith("Poorly differentiated"):
            return "G3"
        if raw.startswith("Undifferentiated"):
            return "G4"
        return "Unknown"

    # Prefer summary grade for 2018+ because it reconciles clinical/pathological grade.
    raw = normalize_space(row.get("Derived Summary Grade 2018 (2018+)"))
    mapping = {"1": "G1", "2": "G2", "3": "G3", "4": "G4"}
    return mapping.get(raw, "Unknown")


def parse_tumor_size_value(x: Any, treat_zero_missing: bool = False) -> float:
    s = normalize_space(x)
    if not s or s.lower() in ["nan", "none"]:
        return np.nan
    s_low = s.lower()
    bad_tokens = ["unknown", "unreasonable", "not applicable", "not consistent", "blank"]
    if any(tok in s_low for tok in bad_tokens):
        return np.nan
    m = re.search(r"\d+", s)
    if not m:
        return np.nan
    val = int(m.group(0))
    # SEER conventions: 999 unknown; 401-989 unknown/unreasonable for this field.
    if val == 999 or 401 <= val <= 989 or val >= 990:
        return np.nan
    if treat_zero_missing and val == 0:
        return np.nan
    if val < 0 or val > 400:
        return np.nan
    return float(val)


def map_tumor_size(row: pd.Series, treat_zero_missing: bool = False) -> float:
    year = as_year(row.get("Year of diagnosis"))
    # Use year-specific source where available, with the over-time recode as fallback.
    candidates: List[Any] = []
    if year is not None and year <= 2015:
        candidates.append(row.get("CS tumor size (2004-2015)"))
    elif year is not None and year >= 2016:
        candidates.append(row.get("Tumor Size Summary (2016+)"))
    candidates.append(row.get("Tumor Size Over Time Recode (1988+)"))
    for v in candidates:
        parsed = parse_tumor_size_value(v, treat_zero_missing=treat_zero_missing)
        if not pd.isna(parsed):
            return parsed
    return np.nan


def coding_era(year: int) -> str:
    if year <= 2015:
        return "AJCC7/CS era (2010-2015)"
    if year in [2016, 2017]:
        return "SEER combined era (2016-2017)"
    return "EOD2018 era (2018+)"


def build_main_cohort(raw: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    required = [
        "Year of diagnosis",
        "SEER Combined Mets at DX-liver (2010+)",
        "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)",
        "Marital status at diagnosis",
        "Diagnostic Confirmation",
        "Derived AJCC T, 7th ed (2010-2015)",
        "Derived SEER Combined T (2016-2017)",
        "Derived EOD 2018 T Recode (2018+)",
        "Derived AJCC N, 7th ed (2010-2015)",
        "Derived SEER Combined N (2016-2017)",
        "Derived EOD 2018 N Recode (2018+)",
        "Grade Recode (thru 2017)",
        "Derived Summary Grade 2018 (2018+)",
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    flow: Dict[str, Any] = {}
    df = raw.copy()
    flow["raw_n"] = int(len(df))

    df["Year_of_diagnosis"] = df["Year of diagnosis"].apply(as_year)
    df = df[df["Year_of_diagnosis"].between(args.dev_years_start, args.test_years_end, inclusive="both")].copy()
    flow["within_year_window_n"] = int(len(df))

    if args.adenocarcinoma_only:
        # Broad adenocarcinoma-family codes. Kept optional because prior pipeline appears to include all gallbladder cancers.
        adeno_codes = set([
            8140, 8141, 8143, 8144, 8210, 8211, 8260, 8261, 8262, 8263,
            8310, 8323, 8480, 8481, 8490, 8500, 8503, 8560, 8570, 8571, 8574,
            8255, 8160, 8163,
        ])
        code_col = "Histologic Type ICD-O-3"
        if code_col not in df.columns:
            raise ValueError("--adenocarcinoma-only requested, but Histologic Type ICD-O-3 column is missing.")
        df["Histology_Code"] = pd.to_numeric(df[code_col], errors="coerce")
        df = df[df["Histology_Code"].isin(adeno_codes)].copy()
        flow["adenocarcinoma_family_n"] = int(len(df))

    df[TARGET_COL] = df["SEER Combined Mets at DX-liver (2010+)"].apply(map_liver_mets)
    df = df[df[TARGET_COL].isin([0, 1])].copy()
    flow["known_liver_mets_n"] = int(len(df))

    collapse_t2 = not args.fine_t2_main
    df["Unified_T"] = df.apply(lambda r: map_t_stage(r, collapse_t2=collapse_t2), axis=1)
    df["Unified_N"] = df.apply(map_n_stage, axis=1)
    df["Unified_Grade"] = df.apply(map_grade, axis=1)
    df["Tumor_Size_Observed"] = df.apply(
        lambda r: map_tumor_size(r, treat_zero_missing=args.treat_zero_size_missing), axis=1
    )
    df["Missing_Size_Indicator"] = df["Tumor_Size_Observed"].isna().astype(int)

    df = df[df["Unified_T"].notna()].copy()
    flow["harmonized_T1_T2_n"] = int(len(df))

    df["Age_Years"] = df["Age recode with <1 year olds and 90+"].apply(parse_age_years)
    df["Age_Group"] = df["Age_Years"].apply(age_group)
    df["Race_Group"] = df["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)"].apply(map_race_origin)
    df["Marital_Status"] = df["Marital status at diagnosis"].apply(map_marital_status)
    df["Histology_Confirmed"] = df["Diagnostic Confirmation"].apply(map_histology_confirmed)
    df["Sex"] = df["Sex"].fillna("Unknown").astype(str).replace({"Blank(s)": "Unknown", "": "Unknown"})
    df["Coding_Era"] = df["Year_of_diagnosis"].astype(int).apply(coding_era)

    exclude_years = parse_year_list(args.exclude_years)
    if exclude_years:
        df = df[~df["Year_of_diagnosis"].isin(exclude_years)].copy()
        flow["after_user_excluded_years_n"] = int(len(df))
        flow["excluded_years"] = sorted(exclude_years)

    def split_label(y: int) -> Optional[str]:
        if args.dev_years_start <= y <= args.dev_years_end:
            return f"Development ({args.dev_years_start}-{args.dev_years_end})"
        if args.test_years_start <= y <= args.test_years_end:
            return f"Temporal Test ({args.test_years_start}-{args.test_years_end})"
        return None

    df["Temporal_Split"] = df["Year_of_diagnosis"].astype(int).apply(split_label)
    df = df[df["Temporal_Split"].notna()].copy()
    flow["analysis_year_split_n"] = int(len(df))

    if "Patient ID" in df.columns:
        df["Patient_ID"] = df["Patient ID"].astype(str)
    else:
        df["Patient_ID"] = [f"P{i:08d}" for i in range(len(df))]

    keep_cols = META_COLS + [TARGET_COL] + [
        "Age_Years", "Age_Group", "Sex", "Race_Group", "Marital_Status",
        "Histology_Confirmed", "Unified_T", "Unified_N", "Unified_Grade",
        "Tumor_Size_Observed", "Missing_Size_Indicator",
        "Year of diagnosis",
        "SEER Combined Mets at DX-liver (2010+)",
        "Marital status at diagnosis",
        "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)",
        "Diagnostic Confirmation",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()

    out["T_NOS"] = out["Unified_T"].astype(str).str.contains("NOS", na=False).astype(int)
    out["N_X"] = (out["Unified_N"] == "NX").astype(int)
    out["Grade_Unknown"] = (out["Unified_Grade"] == "Unknown").astype(int)
    out["Tumor_Size_Missing"] = out["Missing_Size_Indicator"].astype(int)

    return out.reset_index(drop=True), flow


def parse_year_list(x: str) -> set:
    if not x:
        return set()
    out = set()
    for part in str(x).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except Exception:
            pass
    return out


def impute_tumor_size_by_dev_median(df: pd.DataFrame, split_col: str = "Temporal_Split") -> pd.DataFrame:
    df = df.copy()
    dev_mask = df[split_col].astype(str).str.startswith("Development")
    med = df.loc[dev_mask, "Tumor_Size_Observed"].median(skipna=True)
    if pd.isna(med):
        med = df["Tumor_Size_Observed"].median(skipna=True)
    if pd.isna(med):
        med = 25.0
    df["Tumor_Size_Model"] = df["Tumor_Size_Observed"].fillna(float(med))
    df.attrs["tumor_size_dev_median"] = float(med)
    return df


def make_tier_datasets(main: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    main = impute_tumor_size_by_dev_median(main)

    tier1 = main[META_COLS + [TARGET_COL] + TIER1_FEATURES].copy()

    tier3 = main[META_COLS + [TARGET_COL] + TIER3_FEATURES + [
        "Tumor_Size_Observed", "T_NOS", "N_X", "Grade_Unknown", "Tumor_Size_Missing"
    ]].copy()

    t_clear = ~main["Unified_T"].astype(str).str.contains("NOS", na=False)
    n_clear = main["Unified_N"].isin(["N0", "N1", "N2"])
    g_clear = main["Unified_Grade"].isin(["G1", "G2", "G3", "G4"])
    size_clear = main["Tumor_Size_Observed"].notna()
    tier2_base = main[t_clear & n_clear & g_clear & size_clear].copy()
    tier2 = tier2_base[META_COLS + [TARGET_COL] + TIER2_FEATURES + ["Tumor_Size_Observed"]].copy()

    return {"Tier1": tier1, "Tier2": tier2, "Tier3": tier3}


# -----------------------------------------------------------------------------
# Preflight and tables
# -----------------------------------------------------------------------------

def preflight_checks(raw: pd.DataFrame, main: pd.DataFrame, tiers: Dict[str, pd.DataFrame], args: argparse.Namespace) -> List[str]:
    notes: List[str] = []

    raw_years = set(pd.to_numeric(raw["Year of diagnosis"], errors="coerce").dropna().astype(int).unique())
    main_years = set(main["Year_of_diagnosis"].dropna().astype(int).unique())
    if {2016, 2017}.issubset(raw_years) and not {2016, 2017}.issubset(main_years):
        msg = "2016-2017 exist in raw data but are absent from harmonized main cohort. Check SEER Combined T/N mapping."
        if args.allow_missing_2016_2017:
            notes.append("WARNING: " + msg)
        else:
            raise ValueError(msg)

    # Exact marital-status check on raw-to-cleaned mapping.
    if "Marital status at diagnosis" in main.columns:
        bad = main[
            (main["Marital status at diagnosis"] == "Single (never married)")
            & (main["Marital_Status"] == "Married")
        ]
        if len(bad) > 0:
            raise ValueError(f"Marital recoding error: {len(bad)} Single cases mapped to Married.")

    # Race check: no NH American Indian/Unknown should be mapped to Hispanic.
    if "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)" in main.columns:
        bad_race = main[
            main["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)"].isin([
                "Non-Hispanic American Indian/Alaska Native", "Non-Hispanic Unknown Race"
            ]) & (main["Race_Group"] == "Hispanic")
        ]
        if len(bad_race) > 0:
            raise ValueError(f"Race recoding error: {len(bad_race)} non-Hispanic AI/AN/unknown cases mapped to Hispanic.")

    # Tumor size check: 401-989/999 should not survive as observed numeric values.
    if main["Tumor_Size_Observed"].dropna().gt(400).any():
        mx = main["Tumor_Size_Observed"].max()
        raise ValueError(f"Tumor size parsing error: values >400 mm remain; max={mx}")
    if not (main.loc[main["Missing_Size_Indicator"] == 1, "Tumor_Size_Observed"].isna().all()):
        raise ValueError("Missing_Size_Indicator=1 but Tumor_Size_Observed is not NaN for some records.")

    # Coding-era shift warning.
    t_by_era = pd.crosstab(main["Coding_Era"], main["Unified_T"], normalize="index")
    if not t_by_era.empty:
        notes.append("T-stage distribution by coding era was checked. Inspect TableS6/manifest for possible coding-era drift.")

    # Events per tier.
    for tier, df in tiers.items():
        dev = df[df["Temporal_Split"].astype(str).str.startswith("Development")]
        test = df[df["Temporal_Split"].astype(str).str.startswith("Temporal Test")]
        ev_dev = int(dev[TARGET_COL].sum())
        ev_test = int(test[TARGET_COL].sum())
        if ev_dev < 50:
            notes.append(f"WARNING: {tier} has only {ev_dev} development events; boosted-tree tuning will be conservative and results should be interpreted cautiously.")
        if ev_test < 30:
            notes.append(f"WARNING: {tier} has only {ev_test} temporal-test events; performance CIs may be unstable.")

    return notes


def summarize_categorical(series: pd.Series) -> str:
    n = len(series)
    parts = []
    vc = series.fillna("Missing").value_counts(dropna=False)
    for k, v in vc.items():
        pct = 100 * v / n if n else 0
        parts.append(f"{k}: {v} ({pct:.1f}%)")
    return "; ".join(parts)


def summarize_numeric(series: pd.Series) -> str:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0:
        return "NA"
    return f"{x.median():.1f} [{x.quantile(0.25):.1f}, {x.quantile(0.75):.1f}]"



def _format_count_pct(series: pd.Series, level: Any, denom: Optional[int] = None) -> str:
    n = len(series) if denom is None else int(denom)
    cnt = int((series == level).sum())
    pct = 100 * cnt / n if n else 0.0
    return f"{cnt:,} ({pct:.1f})"


def _format_binary_yes(series: pd.Series, positive: Any = 1) -> str:
    n = len(series)
    vals = series.fillna(0)
    cnt = int((vals == positive).sum())
    pct = 100 * cnt / n if n else 0.0
    return f"{cnt:,} ({pct:.1f})"


def make_baseline_table(df: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    """Journal-style baseline table with one row per category.

    Earlier debug versions collapsed multi-category variables into a single long
    cell. That is useful for internal auditing but looks like a program dump in a
    manuscript. This version outputs indented category rows, with cohort size and
    event counts explicitly listed.
    """
    def group_label(x: Any) -> str:
        if group_col == TARGET_COL:
            try:
                return "Liver metastasis" if int(x) == 1 else "No liver metastasis"
            except Exception:
                return str(x)
        return str(x)

    if group_col is None:
        groups = [("Overall", df)]
    else:
        group_items = [(group_label(k), g.copy()) for k, g in df.groupby(group_col, dropna=False)]
        # Stable column ordering.
        if group_col == "Temporal_Split":
            desired = ["Development (2010-2018)", "Temporal Test (2019-2022)"]
            group_items = sorted(group_items, key=lambda kv: desired.index(kv[0]) if kv[0] in desired else 99)
        if group_col == TARGET_COL:
            desired = ["No liver metastasis", "Liver metastasis"]
            group_items = sorted(group_items, key=lambda kv: desired.index(kv[0]) if kv[0] in desired else 99)
        groups = [("Overall", df)] + group_items

    def add_row(rows: List[Dict[str, str]], label: str, func: Callable[[pd.DataFrame], str], section: str = "") -> None:
        row = {"Variable": label}
        if section:
            row["Section"] = section
        for gname, gdf in groups:
            try:
                row[gname] = func(gdf)
            except Exception:
                row[gname] = "NA"
        rows.append(row)

    rows: List[Dict[str, str]] = []
    add_row(rows, "Cohort size, n", lambda g: f"{len(g):,}")
    if TARGET_COL in df.columns:
        add_row(rows, "Liver metastasis events, n (%)", lambda g: _format_binary_yes(g[TARGET_COL], 1))
    if "Age_Years" in df.columns:
        add_row(rows, "Age, years, median [IQR]", lambda g: summarize_numeric(g["Age_Years"]))

    cat_specs = [
        ("Sex", "Sex, n (%)", ["Female", "Male", "Unknown"]),
        ("Race_Group", "Race/ethnicity, n (%)", ["Non-Hispanic White", "Hispanic", "Non-Hispanic Black", "Asian/Pacific Islander", "Other/Unknown"]),
        ("Marital_Status", "Marital status, n (%)", ["Married", "Unmarried/Single", "Divorced/Separated/Widowed", "Unknown"]),
        ("Histology_Confirmed", "Diagnostic confirmation, n (%)", ["Histology confirmed", "Other microscopic confirmation", "Non-microscopic/clinical/unknown"]),
        ("Unified_T", "T subgroup, n (%)", ["T1a", "T1b", "T1NOS", "T2", "T2a", "T2b", "T2NOS"]),
        ("Unified_N", "N subgroup, n (%)", ["N0", "N1", "N2", "NX"]),
        ("Unified_Grade", "Tumor grade, n (%)", ["G1", "G2", "G3", "G4", "Unknown"]),
    ]
    for col, section, levels in cat_specs:
        if col not in df.columns:
            continue
        rows.append({"Variable": section, **{gname: "" for gname, _ in groups}})
        present = [lv for lv in levels if lv in set(df[col].fillna("Unknown").astype(str))]
        for lv in present:
            add_row(rows, f"    {lv}", lambda g, col=col, lv=lv: _format_count_pct(g[col].fillna("Unknown").astype(str), lv), section=section)

    if "Tumor_Size_Observed" in df.columns:
        add_row(rows, "Tumor size, mm, median [IQR]", lambda g: summarize_numeric(g["Tumor_Size_Observed"]))
    if "Missing_Size_Indicator" in df.columns:
        add_row(rows, "Tumor size missing, n (%)", lambda g: _format_binary_yes(g["Missing_Size_Indicator"], 1))
    if "Coding_Era" in df.columns:
        rows.append({"Variable": "Coding era, n (%)", **{gname: "" for gname, _ in groups}})
        eras = ["AJCC7_2010_2015", "SEERCombined_2016_2017", "EOD2018_2018plus"]
        for lv in eras:
            if lv in set(df["Coding_Era"].fillna("Unknown").astype(str)):
                add_row(rows, f"    {lv}", lambda g, lv=lv: _format_count_pct(g["Coding_Era"].fillna("Unknown").astype(str), lv), section="Coding era, n (%)")

    out = pd.DataFrame(rows)
    if "Section" in out.columns:
        out = out.drop(columns=["Section"])
    return out

def predictor_timing_table() -> pd.DataFrame:
    rows = [
        ["Age_Years / Age_Group", "Age recode with <1 year olds and 90+", "Age at diagnosis", "Earliest demographic", "Tier 1/2/3", "Unknown retained if present; continuous plus display group"],
        ["Sex", "Sex", "Biological sex recorded by registry", "Earliest demographic", "Tier 1/2/3", "Unknown retained if present"],
        ["Race_Group", "Race and origin recode", "Race/ethnicity grouping", "Earliest demographic/registry", "Tier 1/2/3", "NH AI/AN and NH unknown race -> Other/Unknown; Hispanic only from race-origin recode"],
        ["Marital_Status", "Marital status at diagnosis", "Marital status", "Diagnosis-period demographic", "Tier 1/2/3", "Exact matching; Single never married is not Married"],
        ["Histology_Confirmed", "Diagnostic Confirmation", "Confirmation method", "Diagnosis-period variable", "Tier 2/3", "Separated into histology, other microscopic, and non-microscopic/clinical/unknown"],
        ["Unified_T", "AJCC7 T / SEER Combined T / EOD2018 T", "Harmonized T subgroup", "Diagnosis-period staging", "Tier 2/3", "Main analysis collapses T2a/T2b into T2 by default; T-NOS retained only in Tier 3"],
        ["Unified_N", "AJCC7 N / SEER Combined N / EOD2018 N", "Harmonized nodal stage", "Diagnosis-period staging", "Tier 2/3", "NX retained only in Tier 3"],
        ["Unified_Grade", "Grade Recode / Derived Summary Grade 2018", "Tumor grade", "Diagnosis-period pathology", "Tier 2/3", "Unknown retained only in Tier 3"],
        ["Tumor_Size_Observed", "CS tumor size / Tumor Size Summary / Tumor Size Over Time", "Observed tumor size in mm", "Diagnosis-period measurement", "Tier 2/3", "401-989, 999, text unknown, blank -> missing"],
        ["Tumor_Size_Model", "Derived from Tumor_Size_Observed", "Median-imputed tumor size for modeling", "Model preprocessing", "Tier 2/3", "Development-median imputation; missingness flag retained only in Tier 3"],
        ["Missing_Size_Indicator", "Derived", "Tumor-size missingness", "Registry uncertainty marker", "Tier 3", "Explicit uncertainty-related feature"],
    ]
    return pd.DataFrame(rows, columns=[
        "Harmonized predictor", "SEER source field(s)", "Clinical meaning",
        "Likely earliest real-world availability", "Assigned tier(s)", "Uncertainty handling"
    ])


def mapping_table() -> pd.DataFrame:
    rows = [
        ["Marital_Status", "Marital status at diagnosis", "Married (including common law)", "Married", "Exact match"],
        ["Marital_Status", "Marital status at diagnosis", "Single (never married)", "Unmarried/Single", "Prevents substring false-positive with 'married'"],
        ["Marital_Status", "Marital status at diagnosis", "Unmarried or Domestic Partner", "Unmarried/Single", "Exact match"],
        ["Marital_Status", "Marital status at diagnosis", "Divorced / Separated / Widowed", "Divorced/Separated/Widowed", "Collapsed clinically similar non-married categories"],
        ["Race_Group", "Race and origin recode", "Non-Hispanic White", "Non-Hispanic White", "Exact from combined race-origin recode"],
        ["Race_Group", "Race and origin recode", "Non-Hispanic Black", "Non-Hispanic Black", "Exact from combined race-origin recode"],
        ["Race_Group", "Race and origin recode", "Hispanic (All Races)", "Hispanic", "Only this source maps to Hispanic"],
        ["Race_Group", "Race and origin recode", "Non-Hispanic Asian or Pacific Islander", "Asian/Pacific Islander", "Exact from combined race-origin recode"],
        ["Race_Group", "Race and origin recode", "NH AI/AN or NH Unknown Race", "Other/Unknown", "Avoids incorrect Hispanic assignment"],
        ["Unified_T", "Derived SEER Combined T (2016-2017)", "c1A/p1A/c1B/p1B/c1/p1/c2/p2", "T1a/T1b/T1NOS/T2", "Includes 2016-2017 instead of silently dropping them"],
        ["Tumor_Size_Observed", "Tumor size fields", "401-989 / 999 / Unknown / Blank", "Missing", "Do not input 901 or 999 as real size"],
    ]
    return pd.DataFrame(rows, columns=["Harmonized variable", "Raw SEER source field", "Raw value", "Mapped value", "Reason / notes"])


def uncertainty_profile(main: pd.DataFrame) -> pd.DataFrame:
    groups = {"Overall": main}
    for name, g in main.groupby("Temporal_Split"):
        groups[name] = g
    rows = []
    for gname, g in groups.items():
        n = len(g)
        rows.append({
            "Cohort": gname,
            "N": n,
            "Events": int(g[TARGET_COL].sum()),
            "Event rate": g[TARGET_COL].mean() if n else np.nan,
            "T-NOS %": 100 * g["T_NOS"].mean() if n else np.nan,
            "NX %": 100 * g["N_X"].mean() if n else np.nan,
            "Grade unknown %": 100 * g["Grade_Unknown"].mean() if n else np.nan,
            "Tumor size missing %": 100 * g["Tumor_Size_Missing"].mean() if n else np.nan,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------

def get_feature_columns(tier: str, df: pd.DataFrame) -> List[str]:
    if tier == "Tier1":
        cols = [c for c in TIER1_FEATURES if c in df.columns]
    elif tier == "Tier2":
        cols = [c for c in TIER2_FEATURES if c in df.columns]
    elif tier == "Tier3":
        cols = [c for c in TIER3_FEATURES if c in df.columns]
    else:
        cols = [c for c in df.columns if c not in META_COLS + [TARGET_COL]]
    return cols


def split_num_cat(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    numeric_cols, categorical_cols = [], []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols


def onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str], spline: bool = False) -> ColumnTransformer:
    if spline and numeric_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False)),
            ("scaler", StandardScaler(with_mean=False)),
        ])
    else:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot_encoder()),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_sklearn_estimator(model: str, numeric_cols: List[str], categorical_cols: List[str], args: argparse.Namespace, scale_pos_weight: float) -> Tuple[Pipeline, Dict[str, Any]]:
    model_l = model.lower()
    if model_l == "linear logistic":
        clf = LogisticRegression(max_iter=3000, solver="lbfgs", class_weight=None)
        pre = make_preprocessor(numeric_cols, categorical_cols, spline=False)
        pipe = Pipeline([("preprocess", pre), ("clf", clf)])
        params = {"clf__C": loguniform(0.05, 20.0)}
        return pipe, params

    if model_l == "spline logistic":
        clf = LogisticRegression(max_iter=3000, solver="lbfgs", class_weight=None)
        pre = make_preprocessor(numeric_cols, categorical_cols, spline=True)
        pipe = Pipeline([("preprocess", pre), ("clf", clf)])
        params = {"clf__C": loguniform(0.05, 20.0)}
        return pipe, params

    if model_l == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed")
        clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=args.xgb_tree_method,
            random_state=args.random_state,
            n_jobs=args.n_jobs_model,
            scale_pos_weight=scale_pos_weight,
        )
        pre = make_preprocessor(numeric_cols, categorical_cols, spline=False)
        pipe = Pipeline([("preprocess", pre), ("clf", clf)])
        # Wider but conservative tree search.  The class-weight parameter is deliberately
        # tunable because using the full negative/positive ratio often improves ranking but
        # can badly distort absolute probabilities and decision-curve behavior.
        spw_low = max(0.5, float(scale_pos_weight) * 0.05)
        spw_high = max(spw_low + 1e-6, float(scale_pos_weight) * 1.20)
        params = {
            "clf__n_estimators": randint(120, 1001),
            "clf__max_depth": randint(1, 6),
            "clf__learning_rate": loguniform(0.005, 0.10),
            "clf__subsample": uniform(0.55, 0.40),
            "clf__colsample_bytree": uniform(0.45, 0.50),
            "clf__min_child_weight": randint(1, 21),
            "clf__reg_alpha": loguniform(0.001, 30.0),
            "clf__reg_lambda": loguniform(0.5, 50.0),
            "clf__gamma": uniform(0.0, 2.0),
            "clf__max_delta_step": randint(0, 6),
            "clf__scale_pos_weight": loguniform(spw_low, spw_high),
        }
        return pipe, params

    raise ValueError(f"Unknown sklearn model: {model}")


def prepare_catboost_frame(df: pd.DataFrame, feature_cols: List[str], numeric_cols: List[str], categorical_cols: List[str], dev_medians: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Dict[str, float], List[int]]:
    x = df[feature_cols].copy()
    medians: Dict[str, float] = {} if dev_medians is None else dict(dev_medians)
    for c in numeric_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
        if c not in medians:
            med = x[c].median(skipna=True)
            if pd.isna(med):
                med = 0.0
            medians[c] = float(med)
        x[c] = x[c].fillna(medians[c]).astype(float)
    for c in categorical_cols:
        x[c] = x[c].astype("object").where(x[c].notna(), "__MISSING__")
        x[c] = x[c].astype(str).replace({"": "__MISSING__", "nan": "__MISSING__", "None": "__MISSING__"})
    cat_indices = [feature_cols.index(c) for c in categorical_cols]
    return x, medians, cat_indices




def _native_scalar(value: Any) -> Any:
    """Convert NumPy scalar values to native Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _native_scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_native_scalar(v) for v in value)
    return value


def sanitize_catboost_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return clone-safe CatBoost parameters.

    Some CatBoost/sklearn combinations fail when RandomizedSearchCV-selected
    SciPy/NumPy scalar parameters are later passed through sklearn.clone().
    Converting to native scalars and recreating CatBoostClassifier avoids that.
    """
    return {k: _native_scalar(v) for k, v in dict(params).items()}


def new_catboost_from_model(model: Any) -> Any:
    """Create a fresh CatBoostClassifier with clone-safe parameters."""
    if not HAS_CAT:
        raise RuntimeError("catboost is not installed")
    params = sanitize_catboost_params(model.get_params())
    return CatBoostClassifier(**params)


def choose_cv(y: np.ndarray, requested_folds: int, random_state: int) -> StratifiedKFold:
    y = np.asarray(y)
    counts = np.bincount(y.astype(int), minlength=2)
    max_folds = max(2, int(min(counts[counts > 0])))
    folds = int(max(2, min(requested_folds, max_folds)))
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)



def _calibration_methods_from_args(args: argparse.Namespace) -> List[str]:
    mode = getattr(args, "tree_calibration", "both")
    if mode == "none":
        return ["none"]
    if mode in ["sigmoid", "isotonic"]:
        return [mode]
    return ["none", "sigmoid", "isotonic"]


def fit_prob_calibrator(y: np.ndarray, p: np.ndarray, method: str) -> Any:
    """Fit a one-dimensional development-only probability calibrator."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if method == "none":
        return {"method": "none"}
    if method == "sigmoid":
        cal = LogisticRegression(max_iter=2000, solver="lbfgs")
        cal.fit(logit_clip(p).reshape(-1, 1), y)
        return {"method": "sigmoid", "model": cal}
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit(np.clip(p, 0, 1), y)
        return {"method": "isotonic", "model": cal}
    raise ValueError(f"Unknown calibration method: {method}")


def apply_prob_calibrator(calibrator: Any, p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if calibrator is None:
        return np.clip(p, 1e-6, 1 - 1e-6)
    method = calibrator.get("method", "none") if isinstance(calibrator, dict) else "none"
    if method == "none":
        return np.clip(p, 1e-6, 1 - 1e-6)
    if method == "sigmoid":
        out = calibrator["model"].predict_proba(logit_clip(p).reshape(-1, 1))[:, 1]
        return np.clip(out, 1e-6, 1 - 1e-6)
    if method == "isotonic":
        out = calibrator["model"].predict(np.clip(p, 0, 1))
        return np.clip(out, 1e-6, 1 - 1e-6)
    return np.clip(p, 1e-6, 1 - 1e-6)


def tree_multi_metric_score(y_true: np.ndarray, p: np.ndarray, args: argparse.Namespace) -> float:
    """Single scalar used for tree HPO when the goal is not one metric only.

    This is not a claim that all metrics can be simultaneously maximized.  It is a
    pre-specified compromise: preserve positive-class ranking (PR-AUC), keep global
    discrimination (ROC-AUC), avoid very poor Brier score, and keep locked-threshold
    F1 acceptable.  It is intentionally development/OOF only.
    """
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return -np.inf
    objective = getattr(args, "tree_objective", "balanced")
    try:
        roc = safe_auc(y, p)
        pr = safe_ap(y, p)
        brier = float(brier_score_loss(y, p))
        thr = select_threshold(y, p, rule=getattr(args, "threshold_rule", "sensitivity80"), target_sens=getattr(args, "target_sensitivity", 0.8))
        f1 = float(f1_score(y, (p >= thr).astype(int), zero_division=0))
        intercept, slope = calibration_intercept_slope(y, p)
        if pd.isna(slope):
            slope_penalty = 0.0
        else:
            slope_penalty = min(1.0, abs(float(slope) - 1.0))
        if objective == "pr_auc":
            return pr
        if objective == "roc_auc":
            return roc
        if objective == "brier":
            return -brier
        if objective == "f1":
            return f1
        # Balanced default.  PR-AUC is weighted highest because this is a low-prevalence
        # metastasis-detection problem; Brier and calibration penalties prevent the
        # boosted trees from winning only by producing poorly calibrated scores.
        return float(2.00 * pr + 0.75 * roc + 0.35 * f1 - 1.25 * brier - 0.15 * slope_penalty)
    except Exception:
        return -np.inf


def best_calibrated_oof_score(y: np.ndarray, raw_p: np.ndarray, args: argparse.Namespace) -> Tuple[float, str, np.ndarray]:
    best_score = -np.inf
    best_method = "none"
    best_pred = np.clip(np.asarray(raw_p, dtype=float), 1e-6, 1 - 1e-6)
    for method in _calibration_methods_from_args(args):
        try:
            cal = fit_prob_calibrator(y, raw_p, method)
            pp = apply_prob_calibrator(cal, raw_p)
            s = tree_multi_metric_score(y, pp, args)
            if s > best_score:
                best_score = s
                best_method = method
                best_pred = pp
        except Exception:
            continue
    return best_score, best_method, best_pred


def apply_development_only_calibration(
    y_dev: np.ndarray,
    raw_oof: np.ndarray,
    raw_test: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, str, Optional[Any], float]:
    """Select and fit a probability calibrator using development OOF predictions only."""
    best_score = -np.inf
    best_method = "none"
    best_cal = fit_prob_calibrator(y_dev, raw_oof, "none")
    best_oof = np.clip(raw_oof, 1e-6, 1 - 1e-6)
    best_test = np.clip(raw_test, 1e-6, 1 - 1e-6)
    for method in _calibration_methods_from_args(args):
        try:
            cal = fit_prob_calibrator(y_dev, raw_oof, method)
            oof_cal = apply_prob_calibrator(cal, raw_oof)
            test_cal = apply_prob_calibrator(cal, raw_test)
            score = tree_multi_metric_score(y_dev, oof_cal, args)
            if score > best_score:
                best_score = score
                best_method = method
                best_cal = cal
                best_oof = oof_cal
                best_test = test_cal
        except Exception as e:
            logging.warning("Calibration method %s failed: %s", method, e)
    return best_oof, best_test, best_method, best_cal, float(best_score)


def manual_oof_for_sklearn(estimator: Pipeline, X: pd.DataFrame, y: np.ndarray, cv: StratifiedKFold) -> np.ndarray:
    pred = np.zeros(len(y), dtype=float)
    for train_idx, valid_idx in cv.split(X, y):
        est = clone(estimator)
        est.fit(X.iloc[train_idx], y[train_idx])
        pred[valid_idx] = np.asarray(est.predict_proba(X.iloc[valid_idx])[:, 1], dtype=float)
    return pred


def tune_sklearn_model(
    model_name: str,
    X_dev: pd.DataFrame,
    y_dev: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    args: argparse.Namespace,
    n_iter: int,
) -> Tuple[Any, Dict[str, Any], Optional[float]]:
    pos = max(1, int(y_dev.sum()))
    neg = max(1, int(len(y_dev) - y_dev.sum()))
    scale_pos_weight = neg / pos
    estimator, param_dist = make_sklearn_estimator(model_name, numeric_cols, categorical_cols, args, scale_pos_weight)
    cv = choose_cv(y_dev, args.cv_folds, args.random_state)

    # For XGBoost, use an explicit OOF loop so the search can optimize a balanced
    # objective instead of a single sklearn scorer.  Logistic baselines remain in
    # RandomizedSearchCV because they are comparators, not the optimization target.
    if model_name.lower() == "xgboost" and n_iter > 1:
        default_candidate = {
            "clf__n_estimators": 260,
            "clf__max_depth": 2,
            "clf__learning_rate": 0.035,
            "clf__subsample": 0.75,
            "clf__colsample_bytree": 0.70,
            "clf__min_child_weight": 3,
            "clf__reg_alpha": 1.0,
            "clf__reg_lambda": 5.0,
            "clf__gamma": 0.1,
            "clf__max_delta_step": 1,
            "clf__scale_pos_weight": max(1.0, scale_pos_weight * 0.35),
        }
        sampler = list(ParameterSampler(param_dist, n_iter=max(0, int(n_iter) - 1), random_state=args.random_state))
        candidates = [default_candidate] + sampler
        best_score = -np.inf
        best_params = default_candidate
        for i, params in enumerate(candidates):
            try:
                est = clone(estimator).set_params(**{k: _native_scalar(v) for k, v in params.items()})
                raw_oof = manual_oof_for_sklearn(est, X_dev, y_dev, cv)
                score, cal_method, _ = best_calibrated_oof_score(y_dev, raw_oof, args)
                if score > best_score:
                    best_score = score
                    best_params = {k: _native_scalar(v) for k, v in params.items()}
            except Exception as e:
                logging.warning("XGBoost candidate %d failed: %s", i, e)
                continue
        final = clone(estimator).set_params(**best_params)
        final.fit(X_dev, y_dev)
        return final, strip_prefix_from_params(best_params), float(best_score) if np.isfinite(best_score) else None

    if n_iter <= 1:
        estimator.fit(X_dev, y_dev)
        return estimator, {}, None
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=args.search_scoring,
        cv=cv,
        random_state=args.random_state,
        n_jobs=args.n_jobs_search,
        refit=True,
        error_score="raise",
    )
    search.fit(X_dev, y_dev)
    return search.best_estimator_, strip_prefix_from_params(search.best_params_), float(search.best_score_)

def sample_catboost_params(rng: np.random.Generator) -> Dict[str, Any]:
    """Random but conservative CatBoost search space, sampled without sklearn wrappers."""
    return {
        "iterations": int(rng.integers(180, 1201)),
        "depth": int(rng.integers(2, 7)),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.005), np.log(0.10)))),
        "l2_leaf_reg": float(np.exp(rng.uniform(np.log(1.0), np.log(80.0)))),
        "random_strength": float(rng.uniform(0.0, 5.0)),
        "bagging_temperature": float(rng.uniform(0.0, 1.0)),
        "border_count": int(rng.integers(32, 255)),
        # Multiplier applied to the development-set neg/pos ratio.  Lower values often
        # improve calibration/Brier while preserving ranking in low-prevalence data.
        "scale_pos_weight_mult": float(np.exp(rng.uniform(np.log(0.05), np.log(1.20)))),
    }


def make_catboost_estimator(params: Dict[str, Any], scale_pos_weight: float, args: argparse.Namespace, seed_offset: int = 0) -> Any:
    if not HAS_CAT:
        raise RuntimeError("catboost is not installed")
    clean = {k: v for k, v in sanitize_catboost_params(params).items() if not str(k).startswith("_")}
    spw_mult = float(clean.pop("scale_pos_weight_mult", 1.0))
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=int(args.random_state + seed_offset),
        verbose=False,
        allow_writing_files=False,
        thread_count=int(args.n_jobs_model),
        scale_pos_weight=float(scale_pos_weight) * spw_mult,
        **clean,
    )


def score_prediction_for_search(y_true: np.ndarray, p: np.ndarray, scoring: str) -> float:
    if scoring == "roc_auc":
        return safe_auc(y_true, p)
    return safe_ap(y_true, p)


def tune_catboost_model(
    X_dev: pd.DataFrame,
    y_dev: np.ndarray,
    feature_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    args: argparse.Namespace,
    n_iter: int,
) -> Tuple[Any, Dict[str, Any], Optional[float], Dict[str, float], List[int]]:
    """Tune CatBoost with a native cross-validation loop.

    Rationale: several CatBoost + scikit-learn version combinations fail under
    RandomizedSearchCV because sklearn.clone or sklearn tag inspection breaks.
    This function never uses RandomizedSearchCV or clone for CatBoost.
    """
    if not HAS_CAT:
        raise RuntimeError("catboost is not installed")

    rng = np.random.default_rng(args.random_state)
    cv = choose_cv(y_dev, args.cv_folds, args.random_state)
    n_iter = max(1, int(n_iter))

    # Include one conservative default candidate so FAST mode remains stable.
    candidates: List[Dict[str, Any]] = [{
        "iterations": 180 if getattr(args, "fast", False) else 260,
        "depth": 3,
        "learning_rate": 0.045,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.6,
        "bagging_temperature": 0.2,
        "border_count": 64,
        "scale_pos_weight_mult": 0.35,
    }]
    for _ in range(max(0, n_iter - 1)):
        candidates.append(sample_catboost_params(rng))

    best_score = -np.inf
    best_params: Dict[str, Any] = sanitize_catboost_params(candidates[0])

    for i, params in enumerate(candidates):
        oof_pred = np.zeros(len(y_dev), dtype=float)
        ok = True
        for fold_id, (train_idx, valid_idx) in enumerate(cv.split(X_dev, y_dev)):
            X_tr = X_dev.iloc[train_idx].copy()
            X_va = X_dev.iloc[valid_idx].copy()
            y_tr = y_dev[train_idx]
            pos = max(1, int(y_tr.sum()))
            neg = max(1, int(len(y_tr) - y_tr.sum()))
            scale_pos_weight = neg / pos
            try:
                X_tr_cb, med, cats = prepare_catboost_frame(X_tr, feature_cols, numeric_cols, categorical_cols)
                X_va_cb, _, _ = prepare_catboost_frame(X_va, feature_cols, numeric_cols, categorical_cols, dev_medians=med)
                est = make_catboost_estimator(params, scale_pos_weight, args, seed_offset=1000 + i * 17 + fold_id)
                est.fit(X_tr_cb, y_tr, cat_features=cats)
                oof_pred[valid_idx] = np.asarray(est.predict_proba(X_va_cb)[:, 1], dtype=float)
            except Exception as e:
                logging.warning("CatBoost candidate %d fold %d failed: %s", i, fold_id, e)
                ok = False
                break
        if not ok:
            continue
        mean_score, cal_method, _ = best_calibrated_oof_score(y_dev, oof_pred, args)
        if mean_score > best_score:
            best_score = float(mean_score)
            best_params = sanitize_catboost_params(params)
            best_params["_search_selected_calibration"] = cal_method

    X_cb, medians, cat_indices = prepare_catboost_frame(X_dev, feature_cols, numeric_cols, categorical_cols)
    pos_all = max(1, int(y_dev.sum()))
    neg_all = max(1, int(len(y_dev) - y_dev.sum()))
    final = make_catboost_estimator(best_params, neg_all / pos_all, args, seed_offset=9999)
    final.fit(X_cb, y_dev, cat_features=cat_indices)
    return final, best_params, float(best_score) if np.isfinite(best_score) else None, medians, cat_indices

def predict_proba_model(model: Any, X: pd.DataFrame, model_name: str, feature_cols: List[str], numeric_cols: List[str], categorical_cols: List[str], cat_medians: Optional[Dict[str, float]] = None) -> np.ndarray:
    if model_name.lower() == "catboost":
        X_cb, _, _ = prepare_catboost_frame(X, feature_cols, numeric_cols, categorical_cols, dev_medians=cat_medians)
        return np.asarray(model.predict_proba(X_cb)[:, 1], dtype=float)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def oof_predict_model(model: Any, X: pd.DataFrame, y: np.ndarray, model_name: str, feature_cols: List[str], numeric_cols: List[str], categorical_cols: List[str], args: argparse.Namespace, cat_medians: Optional[Dict[str, float]] = None, cat_indices: Optional[List[int]] = None) -> np.ndarray:
    cv = choose_cv(y, args.cv_folds, args.random_state)
    if model_name.lower() != "catboost":
        return cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba", n_jobs=args.n_jobs_search)[:, 1]

    preds = np.zeros(len(y), dtype=float)
    for train_idx, valid_idx in cv.split(X, y):
        X_tr = X.iloc[train_idx].copy()
        X_va = X.iloc[valid_idx].copy()
        y_tr = y[train_idx]
        est = new_catboost_from_model(model)
        X_tr_cb, med, cats = prepare_catboost_frame(X_tr, feature_cols, numeric_cols, categorical_cols)
        X_va_cb, _, _ = prepare_catboost_frame(X_va, feature_cols, numeric_cols, categorical_cols, dev_medians=med)
        est.fit(X_tr_cb, y_tr, cat_features=cats)
        preds[valid_idx] = est.predict_proba(X_va_cb)[:, 1]
    return preds


def select_threshold(y: np.ndarray, p: np.ndarray, rule: str = "sensitivity80", target_sens: float = 0.80) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    candidates = np.unique(np.clip(p, 0, 1))
    if len(candidates) == 0:
        return 0.5
    rows = []
    for t in candidates:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        youden = sens + spec - 1 if not np.isnan(sens + spec) else -np.inf
        rows.append((t, sens, spec, youden))
    met = pd.DataFrame(rows, columns=["threshold", "sensitivity", "specificity", "youden"])
    if rule.lower() in ["sensitivity80", "sensitivity_target", "sens80"]:
        ok = met[met["sensitivity"] >= target_sens]
        if len(ok):
            ok = ok.sort_values(["specificity", "threshold"], ascending=[False, False])
            return float(ok.iloc[0]["threshold"])
    # fallback Youden
    met = met.sort_values(["youden", "threshold"], ascending=[False, False])
    return float(met.iloc[0]["threshold"])


def compute_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    out = {
        "N": len(y),
        "Events": int(y.sum()),
        "Event rate": float(y.mean()) if len(y) else np.nan,
        "ROC-AUC": safe_auc(y, p),
        "PR-AUC": safe_ap(y, p),
        "Brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV": ppv,
        "NPV": npv,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "H-L P value": hosmer_lemeshow_p(y, p),
        "Calibration intercept": np.nan,
        "Calibration slope": np.nan,
    }
    intercept, slope = calibration_intercept_slope(y, p)
    out["Calibration intercept"] = intercept
    out["Calibration slope"] = slope
    return out


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    if not HAS_SM or len(np.unique(y)) < 2:
        return np.nan, np.nan
    try:
        lp = logit_clip(p)
        X = sm.add_constant(lp)
        fit = sm.Logit(y, X).fit(disp=False, maxiter=200)
        return float(fit.params[0]), float(fit.params[1])
    except Exception:
        return np.nan, np.nan


def hosmer_lemeshow_p(y: np.ndarray, p: np.ndarray, g: int = 10) -> float:
    try:
        y = np.asarray(y).astype(int)
        p = np.asarray(p, dtype=float)
        if len(np.unique(y)) < 2:
            return np.nan
        df = pd.DataFrame({"y": y, "p": p})
        df["bin"] = pd.qcut(df["p"], q=min(g, df["p"].nunique()), duplicates="drop")
        obs = df.groupby("bin", observed=True)["y"].sum()
        exp = df.groupby("bin", observed=True)["p"].sum()
        n = df.groupby("bin", observed=True)["y"].size()
        stat = (((obs - exp) ** 2) / (exp + 1e-12) + (((n - obs) - (n - exp)) ** 2) / ((n - exp) + 1e-12)).sum()
        dof = max(1, len(obs) - 2)
        return float(1 - stats.chi2.cdf(stat, dof))
    except Exception:
        return np.nan


def bootstrap_ci(y: np.ndarray, p: np.ndarray, threshold: float, metric: str, n_boot: int, random_state: int) -> Tuple[float, float]:
    rng = np.random.default_rng(random_state)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        try:
            if metric == "ROC-AUC":
                vals.append(roc_auc_score(yy, pp))
            elif metric == "PR-AUC":
                vals.append(average_precision_score(yy, pp))
            elif metric == "Brier":
                vals.append(brier_score_loss(yy, pp))
            elif metric == "F1":
                vals.append(f1_score(yy, (pp >= threshold).astype(int), zero_division=0))
        except Exception:
            continue
    if len(vals) < 10:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def format_ci(point: float, lo: float, hi: float, digits: int = 3) -> str:
    if pd.isna(point):
        return "NA"
    if pd.isna(lo) or pd.isna(hi):
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def run_one_model(tier: str, model_name: str, df: pd.DataFrame, args: argparse.Namespace) -> ModelResult:
    feature_cols = get_feature_columns(tier, df)
    numeric_cols, categorical_cols = split_num_cat(df, feature_cols)
    dev_mask = df["Temporal_Split"].astype(str).str.startswith("Development")
    test_mask = df["Temporal_Split"].astype(str).str.startswith("Temporal Test")
    dev = df[dev_mask].copy()
    test = df[test_mask].copy()
    X_dev = dev[feature_cols].copy()
    y_dev = dev[TARGET_COL].astype(int).to_numpy()
    X_test = test[feature_cols].copy()
    y_test = test[TARGET_COL].astype(int).to_numpy()

    if len(dev) == 0 or len(test) == 0:
        raise ValueError(f"{tier}: empty development or temporal test set.")
    if len(np.unique(y_dev)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError(f"{tier}: target has fewer than 2 classes in development or test set.")

    warnings_list: List[str] = []
    dev_events = int(y_dev.sum())
    if dev_events < 50 and model_name.lower() in ["xgboost", "catboost"]:
        warnings_list.append(f"{tier} has {dev_events} development events; search iterations reduced.")
        n_iter = min(5, max(1, args.search_iters_xgb if model_name.lower() == "xgboost" else args.search_iters_cat))
    else:
        if model_name.lower() == "xgboost":
            n_iter = args.search_iters_xgb
        elif model_name.lower() == "catboost":
            n_iter = args.search_iters_cat
        else:
            n_iter = args.search_iters_logistic

    if args.fast:
        n_iter = min(n_iter, 2)

    cat_medians = None
    cat_indices = None
    if model_name.lower() == "catboost":
        estimator, params, cv_score, cat_medians, cat_indices = tune_catboost_model(
            X_dev, y_dev, feature_cols, numeric_cols, categorical_cols, args, n_iter=n_iter
        )
        raw_oof = oof_predict_model(estimator, X_dev, y_dev, model_name, feature_cols, numeric_cols, categorical_cols, args, cat_medians, cat_indices)
        raw_test_pred = predict_proba_model(estimator, X_test, model_name, feature_cols, numeric_cols, categorical_cols, cat_medians)
        oof, test_pred, cal_method, cal_obj, cal_score = apply_development_only_calibration(y_dev, raw_oof, raw_test_pred, args)
        selected_params = dict(params)
        selected_params["_cat_medians"] = cat_medians
        selected_params["_cat_indices"] = cat_indices
        selected_params["probability_calibration"] = cal_method
        selected_params["tree_balanced_oof_score"] = cal_score
        warnings_list.append(f"{model_name}: development-only probability calibration = {cal_method}.")
    else:
        estimator, params, cv_score = tune_sklearn_model(
            model_name, X_dev, y_dev, numeric_cols, categorical_cols, args, n_iter=n_iter
        )
        raw_oof = oof_predict_model(estimator, X_dev, y_dev, model_name, feature_cols, numeric_cols, categorical_cols, args)
        raw_test_pred = predict_proba_model(estimator, X_test, model_name, feature_cols, numeric_cols, categorical_cols)
        selected_params = params
        if model_name.lower() == "xgboost":
            oof, test_pred, cal_method, cal_obj, cal_score = apply_development_only_calibration(y_dev, raw_oof, raw_test_pred, args)
            selected_params = dict(selected_params)
            selected_params["probability_calibration"] = cal_method
            selected_params["tree_balanced_oof_score"] = cal_score
            warnings_list.append(f"{model_name}: development-only probability calibration = {cal_method}.")
        else:
            oof = raw_oof
            test_pred = raw_test_pred

    threshold = select_threshold(y_dev, oof, rule=args.threshold_rule, target_sens=args.target_sensitivity)
    return ModelResult(
        tier=tier, model=model_name, estimator=estimator, threshold=threshold,
        threshold_rule=args.threshold_rule, oof_pred=oof, test_pred=test_pred,
        y_dev=y_dev, y_test=y_test, x_dev=X_dev, x_test=X_test,
        feature_cols=feature_cols, numeric_cols=numeric_cols, categorical_cols=categorical_cols,
        selected_params=selected_params, cv_score=cv_score, warnings=warnings_list,
    )


def run_all_models(tiers: Dict[str, pd.DataFrame], args: argparse.Namespace) -> Tuple[List[ModelResult], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_names = ["Linear Logistic", "Spline Logistic"]
    if HAS_XGB and not args.skip_xgboost:
        model_names.append("XGBoost")
    if HAS_CAT and not args.skip_catboost:
        model_names.append("CatBoost")

    results: List[ModelResult] = []
    pred_rows: List[Dict[str, Any]] = []
    table_rows: List[Dict[str, Any]] = []
    hp_rows: List[Dict[str, Any]] = []

    for tier, df in tiers.items():
        logging.info("Running models for %s (n=%s, events=%s)", tier, len(df), int(df[TARGET_COL].sum()))
        test_meta = df[df["Temporal_Split"].astype(str).str.startswith("Temporal Test")][META_COLS + [TARGET_COL]].reset_index(drop=True)
        for model_name in model_names:
            try:
                logging.info("  Fitting %s / %s", tier, model_name)
                res = run_one_model(tier, model_name, df, args)
                results.append(res)
                metrics = compute_metrics(res.y_test, res.test_pred, res.threshold)
                ci = {}
                for metric in ["ROC-AUC", "PR-AUC", "F1", "Brier"]:
                    ci[metric] = bootstrap_ci(
                        res.y_test, res.test_pred, res.threshold, metric,
                        n_boot=args.bootstrap, random_state=args.random_state + len(results) * 17,
                    )

                row = {
                    "Tier": tier,
                    "Model": model_name,
                    "N test": metrics["N"],
                    "Events test": metrics["Events"],
                    "Event rate test": metrics["Event rate"],
                    "ROC-AUC": metrics["ROC-AUC"],
                    "ROC-AUC 95% CI": format_ci(metrics["ROC-AUC"], *ci["ROC-AUC"]),
                    "PR-AUC": metrics["PR-AUC"],
                    "PR-AUC 95% CI": format_ci(metrics["PR-AUC"], *ci["PR-AUC"]),
                    "F1": metrics["F1"],
                    "F1 95% CI": format_ci(metrics["F1"], *ci["F1"]),
                    "Brier": metrics["Brier"],
                    "Brier 95% CI": format_ci(metrics["Brier"], *ci["Brier"]),
                    "H-L P value": metrics["H-L P value"],
                    "Calibration intercept": metrics["Calibration intercept"],
                    "Calibration slope": metrics["Calibration slope"],
                    "Locked threshold": res.threshold,
                    "Threshold rule": res.threshold_rule,
                    "Sensitivity": metrics["Sensitivity"],
                    "Specificity": metrics["Specificity"],
                    "PPV": metrics["PPV"],
                    "NPV": metrics["NPV"],
                    "TP": metrics["TP"], "FP": metrics["FP"], "TN": metrics["TN"], "FN": metrics["FN"],
                    "Warnings": " | ".join(res.warnings),
                }
                table_rows.append(row)
                hp_rows.append({
                    "Tier": tier,
                    "Model": model_name,
                    "Selected parameters": json.dumps(sanitize_json(res.selected_params), ensure_ascii=False),
                    "Best CV score": res.cv_score,
                    "Search scoring": args.search_scoring,
                    "Feature columns": ", ".join(res.feature_cols),
                })
                for i, p in enumerate(res.test_pred):
                    meta = test_meta.iloc[i].to_dict()
                    pred_rows.append({
                        **meta,
                        "Tier": tier,
                        "Model": model_name,
                        "y_true": int(res.y_test[i]),
                        "pred_prob": float(p),
                        "locked_threshold": float(res.threshold),
                        "pred_class": int(p >= res.threshold),
                    })
            except Exception as e:
                logging.exception("Model failed: %s / %s", tier, model_name)
                table_rows.append({
                    "Tier": tier, "Model": model_name, "Error": repr(e)
                })

    return results, pd.DataFrame(pred_rows), pd.DataFrame(table_rows), pd.DataFrame(hp_rows)


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# -----------------------------------------------------------------------------
# DeLong ROC test
# -----------------------------------------------------------------------------

def compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> Tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) != 2:
        return np.nan
    order = np.argsort(-y_true)
    preds = np.vstack([p1, p2])[:, order]
    label_1_count = int(y_true.sum())
    aucs, cov = fast_delong(preds, label_1_count)
    cov = np.atleast_2d(cov)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0 or pd.isna(var):
        return np.nan
    z = abs(diff) / math.sqrt(var)
    return float(2 * (1 - stats.norm.cdf(z)))


def make_boosted_tree_head_to_head(pred_df: pd.DataFrame, table3: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pred_df.empty:
        return pd.DataFrame()
    for tier in sorted(pred_df["Tier"].unique()):
        sub = pred_df[pred_df["Tier"] == tier]
        models = set(sub["Model"].unique())
        if not {"XGBoost", "CatBoost"}.issubset(models):
            continue
        xgb = sub[sub["Model"] == "XGBoost"].sort_values("Patient_ID")
        cat = sub[sub["Model"] == "CatBoost"].sort_values("Patient_ID")
        merged = xgb[["Patient_ID", "y_true", "pred_prob"]].merge(
            cat[["Patient_ID", "pred_prob"]], on="Patient_ID", suffixes=("_xgb", "_cat")
        )
        p_delong = delong_roc_test(merged["y_true"].to_numpy(), merged["pred_prob_xgb"].to_numpy(), merged["pred_prob_cat"].to_numpy())
        xgb_row = table3[(table3["Tier"] == tier) & (table3["Model"] == "XGBoost")]
        cat_row = table3[(table3["Tier"] == tier) & (table3["Model"] == "CatBoost")]
        if len(xgb_row) and len(cat_row):
            xgb_r = xgb_row.iloc[0]
            cat_r = cat_row.iloc[0]
            winner = "XGBoost" if xgb_r.get("PR-AUC", -np.inf) >= cat_r.get("PR-AUC", -np.inf) else "CatBoost"
            rows.append({
                "Tier": tier,
                "Boosted-tree winner within tier": winner,
                "XGBoost ROC-AUC": xgb_r.get("ROC-AUC"),
                "CatBoost ROC-AUC": cat_r.get("ROC-AUC"),
                "XGBoost PR-AUC": xgb_r.get("PR-AUC"),
                "CatBoost PR-AUC": cat_r.get("PR-AUC"),
                "XGBoost F1": xgb_r.get("F1"),
                "CatBoost F1": cat_r.get("F1"),
                "XGBoost Brier": xgb_r.get("Brier"),
                "CatBoost Brier": cat_r.get("Brier"),
                "DeLong P value: CatBoost vs XGBoost": p_delong,
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Selection, ablation, coefficients
# -----------------------------------------------------------------------------

def select_main_model(results: List[ModelResult], table3: pd.DataFrame, args: argparse.Namespace) -> Optional[ModelResult]:
    t3 = table3[(table3["Tier"] == "Tier3") & (~table3.get("ROC-AUC", pd.Series(dtype=float)).isna())].copy()
    if t3.empty:
        return None
    # Guard against extremely poor calibration if available. If all fail, fall back to all.
    cal_ok = t3[
        t3["Calibration slope"].between(args.cal_slope_min, args.cal_slope_max, inclusive="both")
        | t3["Calibration slope"].isna()
    ]
    pool = cal_ok if len(cal_ok) else t3
    if getattr(args, "selection_strategy", "primary_metric") == "balanced":
        # Conservative manuscript-facing selection: favor PR-AUC/ROC-AUC but penalize
        # poor Brier.  This avoids selecting a model that ranks well but is unusable
        # as a risk-stratification score.
        pool = pool.copy()
        pool["_selection_score"] = (
            2.00 * pool["PR-AUC"].astype(float)
            + 0.75 * pool["ROC-AUC"].astype(float)
            + 0.35 * pool["F1"].astype(float)
            - 1.25 * pool["Brier"].astype(float)
        )
        pool = pool.sort_values(["_selection_score", "PR-AUC", "Brier"], ascending=[False, False, True])
    else:
        metric = args.selection_metric
        if metric not in pool.columns:
            metric = "PR-AUC"
        pool = pool.sort_values([metric, "Brier"], ascending=[False, True])
    chosen_model = str(pool.iloc[0]["Model"])
    for r in results:
        if r.tier == "Tier3" and r.model == chosen_model:
            return r
    return None


def make_risk_table(y: np.ndarray, p: np.ndarray, threshold: float, low_factor: float = 0.5) -> pd.DataFrame:
    low_cut = max(0.0, threshold * low_factor)
    groups = pd.cut(
        p,
        bins=[-np.inf, low_cut, threshold, np.inf],
        labels=["Low", "Intermediate", "High"],
        right=False,
    )
    df = pd.DataFrame({"Risk group": groups.astype(str), "y": y, "p": p})
    total_events = max(1, int(df["y"].sum()))
    rows = []
    for g in ["Low", "Intermediate", "High"]:
        sub = df[df["Risk group"] == g]
        n = len(sub)
        ev = int(sub["y"].sum()) if n else 0
        rows.append({
            "Risk group": g,
            "Lower probability boundary": 0.0 if g == "Low" else (low_cut if g == "Intermediate" else threshold),
            "Upper probability boundary": low_cut if g == "Low" else (threshold if g == "Intermediate" else 1.0),
            "Number of patients": n,
            "Number of liver metastasis events": ev,
            "Observed event rate": ev / n if n else np.nan,
            "Event capture percentage": 100 * ev / total_events,
        })
    return pd.DataFrame(rows)


def make_operating_characteristics(selected: ModelResult) -> pd.DataFrame:
    m = compute_metrics(selected.y_test, selected.test_pred, selected.threshold)
    keys = ["N", "Events", "Event rate", "ROC-AUC", "PR-AUC", "Brier", "F1", "Sensitivity", "Specificity", "PPV", "NPV", "TP", "FP", "TN", "FN", "H-L P value", "Calibration intercept", "Calibration slope"]
    row = {"Tier": selected.tier, "Model": selected.model, "Locked threshold": selected.threshold, "Threshold rule": selected.threshold_rule}
    for k in keys:
        row[k] = m.get(k)
    return pd.DataFrame([row])


def create_uncertainty_masked_tier3(tier3: pd.DataFrame) -> pd.DataFrame:
    df = tier3.copy()
    # Fixed-reference masking: removes uncertainty-category signal while preserving same patients.
    # T-NOS collapsed to broad T category; T2 subdivisions collapsed to T2.
    def mask_t(x: Any) -> str:
        s = str(x)
        if s.startswith("T1"):
            return "T1"
        if s.startswith("T2"):
            return "T2"
        return s
    df["Unified_T"] = df["Unified_T"].apply(mask_t)
    # NX is fixed to N0 reference to remove explicit NX signal.
    df["Unified_N"] = df["Unified_N"].replace({"NX": "N0"})
    # Unknown grade fixed to modal non-unknown grade from the same cohort.
    non_unknown = df.loc[df["Unified_Grade"].isin(["G1", "G2", "G3", "G4"]), "Unified_Grade"]
    modal_grade = non_unknown.mode().iloc[0] if len(non_unknown.mode()) else "G2"
    df["Unified_Grade"] = df["Unified_Grade"].replace({"Unknown": modal_grade})
    if "Missing_Size_Indicator" in df.columns:
        df["Missing_Size_Indicator"] = 0
    return df


def run_same_cohort_ablation(tier3: pd.DataFrame, selected_model_name: str, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    analyses = {
        "Full Tier 3": tier3,
        "Uncertainty-masked same-cohort Tier 3": create_uncertainty_masked_tier3(tier3),
    }
    for analysis, df in analyses.items():
        try:
            res = run_one_model("Tier3", selected_model_name, df, args)
            m = compute_metrics(res.y_test, res.test_pred, res.threshold)
            rows.append({
                "Analysis": analysis,
                "Model": selected_model_name,
                "Cohort N": len(df),
                "Development events": int(res.y_dev.sum()),
                "Temporal-test events": int(res.y_test.sum()),
                "ROC-AUC": m["ROC-AUC"],
                "PR-AUC": m["PR-AUC"],
                "F1": m["F1"],
                "Brier": m["Brier"],
                "H-L P value": m["H-L P value"],
                "Locked threshold": res.threshold,
                "Sensitivity": m["Sensitivity"],
                "Specificity": m["Specificity"],
                "PPV": m["PPV"],
                "NPV": m["NPV"],
            })
        except Exception as e:
            rows.append({"Analysis": analysis, "Model": selected_model_name, "Error": repr(e)})
    return pd.DataFrame(rows)


def logistic_coefficients(results: List[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        if r.model != "Linear Logistic":
            continue
        try:
            pipe = r.estimator
            pre = pipe.named_steps["preprocess"]
            clf = pipe.named_steps["clf"]
            names = pre.get_feature_names_out()
            coefs = clf.coef_.ravel()
            for name, coef in zip(names, coefs):
                if abs(coef) > 20:
                    or_val = np.nan
                else:
                    or_val = float(np.exp(coef))
                rows.append({
                    "Tier": r.tier,
                    "Feature": name,
                    "Coefficient": float(coef),
                    "Odds ratio (model scale)": or_val,
                    "Note": "Sklearn penalized logistic coefficient after preprocessing; categorical coefficients are relative to the dropped one-hot reference level.",
                })
        except Exception as e:
            rows.append({"Tier": r.tier, "Feature": "ERROR", "Note": repr(e)})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------



def draw_box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, label: Optional[str] = None,
             edge: str = None, face: str = "white", text_color: str = None, lw: float = 1.0,
             fontsize: int = 10, bold_first: bool = False, dashed: bool = False, zorder: int = 5,
             rounding: float = 0.012) -> None:
    """Draw a rounded text box in axes coordinates with a label outside the text area."""
    from matplotlib.patches import FancyBboxPatch
    edge = edge or PUB["blue"]
    text_color = text_color or PUB["text"]
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.010,rounding_size={rounding}",
        facecolor=face, edgecolor=edge, linewidth=lw,
        linestyle=(0, (4, 2)) if dashed else "-",
        transform=ax.transAxes, zorder=zorder, clip_on=False
    )
    ax.add_patch(patch)
    ax.text(
        x + w/2, y + h/2, text,
        ha="center", va="center", fontsize=fontsize, color=text_color,
        transform=ax.transAxes, linespacing=1.28,
        fontweight="bold" if bold_first else "normal", zorder=zorder + 1,
        clip_on=False
    )
    if label:
        ax.text(
            x - 0.014, y + h - 0.006, label,
            ha="center", va="center", fontsize=8.4, fontweight="bold",
            color="white", transform=ax.transAxes, clip_on=False, zorder=zorder + 2,
            bbox=dict(boxstyle="round,pad=0.16", facecolor=PUB["navy"], edgecolor=PUB["navy"], linewidth=0)
        )


def arrow_axes(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float, color: str = None, lw: float = 1.0) -> None:
    """Draw connector arrows with endpoints supplied at box boundaries.

    The caller should pass coordinates that already stop at the box edge. We use
    no shrink so the arrow is predictable and does not drift into the text area.
    """
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1), xycoords=ax.transAxes, textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color or PUB["navy"], shrinkA=0, shrinkB=0, mutation_scale=9),
        zorder=3, clip_on=False
    )

def _split_count(df: pd.DataFrame, token: str) -> int:
    if df is None or df.empty or "Temporal_Split" not in df.columns:
        return 0
    s = df["Temporal_Split"].astype(str).str.lower()
    return int(s.str.contains(token.lower(), regex=False).sum())



def save_flowchart(flow: Dict[str, Any], tiers: Dict[str, pd.DataFrame], outpath: Path) -> None:
    """Figure 1: cohort assembly flowchart, v2 layout.

    Design decisions:
    - No embedded figure title; manuscript caption supplies Figure 1 title.
    - Primary branch is vertical and central.
    - Sensitivity-analysis branch is a separate dashed card on the right with no
      overlapping boxes or connector lines.
    - Arrows end exactly at box boundaries and are center-aligned.
    - PDF/SVG/PNG/TIFF are exported by save_pub_figure().
    """
    fig, ax = plt.subplots(figsize=(9.2, 11.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    raw_n = fmt_n(flow.get("raw_n", np.nan))
    known_n = fmt_n(flow.get("known_liver_mets_n", np.nan))
    main_n = fmt_n(flow.get("harmonized_T1_T2_n", np.nan))
    tier3 = tiers.get("Tier3", pd.DataFrame())
    dev_n = fmt_n(_split_count(tier3, "Development"))
    test_n = fmt_n(_split_count(tier3, "Temporal Test"))

    # Central screening flow: all same x-center and arrow endpoints on boundaries.
    x0, w0 = 0.31, 0.38
    stages = [
        (0.875, 0.070, f"Raw SEER extract\nGallbladder cancer cases\nn = {raw_n}", "A"),
        (0.755, 0.060, f"Known liver metastasis status\nn = {known_n}", "B"),
        (0.635, 0.060, f"Harmonized T1–T2 main cohort\nn = {main_n}", "C"),
    ]
    for y, h, txt, lab in stages:
        draw_box(ax, x0, y, w0, h, txt, lab, fontsize=10.2, zorder=6)
    # vertical connectors; stop before arrow head touches the next box text.
    arrow_axes(ax, 0.50, 0.875, 0.50, 0.825)
    arrow_axes(ax, 0.50, 0.755, 0.50, 0.695)

    # Split node from main cohort to primary pathway and sensitivity side-card.
    # This connector deliberately runs in open space only.
    ax.plot([0.50, 0.50], [0.635, 0.598], color=PUB["navy"], lw=1.0, transform=ax.transAxes, zorder=2, clip_on=False)
    ax.plot([0.31, 0.79], [0.598, 0.598], color=PUB["navy"], lw=1.0, transform=ax.transAxes, zorder=2, clip_on=False)
    arrow_axes(ax, 0.31, 0.598, 0.31, 0.570)
    arrow_axes(ax, 0.79, 0.598, 0.79, 0.570)

    # Sensitivity branch as one independent dashed card; never overlaps another box.
    sens_text = "Sensitivity-analysis cohorts\n\nC1  No-NOS cohort: n = 2,555\nC2  Strict cohort: n = 2,011\nC3  Complete-case cohort: n = 1,737"
    draw_box(ax, 0.61, 0.410, 0.35, 0.155, sens_text, None, edge="#8FA6BC", face="#FBFDFF",
             dashed=True, fontsize=8.5, lw=1.0, zorder=4)

    # Primary modeling branch.
    draw_box(ax, 0.12, 0.505, 0.38, 0.064, "Temporal split for model\ndevelopment and validation", "D", fontsize=9.6, zorder=6)
    ax.plot([0.31, 0.31], [0.505, 0.472], color=PUB["navy"], lw=1.0, transform=ax.transAxes, zorder=2, clip_on=False)
    ax.plot([0.18, 0.44], [0.472, 0.472], color=PUB["navy"], lw=1.0, transform=ax.transAxes, zorder=2, clip_on=False)
    arrow_axes(ax, 0.18, 0.472, 0.18, 0.438)
    arrow_axes(ax, 0.44, 0.472, 0.44, 0.438)
    draw_box(ax, 0.045, 0.350, 0.27, 0.082, f"Development cohort\n2010–2018\nn = {dev_n}", "D1", fontsize=9.4, zorder=6)
    draw_box(ax, 0.355, 0.350, 0.27, 0.082, f"Temporal test cohort\n2019–2022\nn = {test_n}", "D2", fontsize=9.4, zorder=6)

    # The tier boxes are linked by a single horizontal bus placed safely above boxes.
    tier_y, tier_h, tier_w = 0.178, 0.083, 0.255
    bus_y = 0.310
    ax.plot([0.18, 0.82], [bus_y, bus_y], color=PUB["navy"], lw=1.0, transform=ax.transAxes, zorder=2, clip_on=False)
    for xcenter in [0.17, 0.50, 0.83]:
        arrow_axes(ax, xcenter, bus_y, xcenter, tier_y + tier_h + 0.004)
    draw_box(ax, 0.045, tier_y, tier_w, tier_h, "Tier 1\nEarliest-available\ndemographic information", "E1", fontsize=8.5, zorder=6)
    draw_box(ax, 0.372, tier_y, tier_w, tier_h, "Tier 2\nPurity-restricted\ndiagnosis-period information", "E2", fontsize=8.5, zorder=6)
    draw_box(ax, 0.700, tier_y, tier_w, tier_h, "Tier 3\nUncertainty-enriched\ndiagnosis-period information", "E3", edge=PUB["orange"], face="#FFF3EA", lw=1.35, fontsize=8.5, zorder=6)
    ax.text(0.930, tier_y + tier_h*0.69, "★", color=PUB["orange"], fontsize=16.5, transform=ax.transAxes, ha="center", va="center", zorder=8)
    ax.text(0.828, 0.145, "Primary analysis tier", color="white", fontsize=7.7, fontweight="bold",
            transform=ax.transAxes, ha="center", va="center", zorder=8,
            bbox=dict(boxstyle="round,pad=0.22", facecolor=PUB["orange"], edgecolor=PUB["orange"], linewidth=0))

    # Footer note as its own wide box.
    footer = "Primary objective: time-validated risk stratification for liver metastasis at diagnosis in T1–T2 gallbladder cancer."
    ax.text(0.50, 0.065, footer, ha="center", va="center", fontsize=8.6, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.48,rounding_size=0.015", facecolor="white", edgecolor="#B8C4D0", linewidth=0.9))

    save_pub_figure(fig, outpath)
    plt.close(fig)


def save_schema_figure(args: argparse.Namespace, outpath: Path) -> None:
    """Figure 2: temporal validation and information hierarchy, v2 layout."""
    from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
    fig = plt.figure(figsize=(13.4, 8.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.83, 1.17], hspace=0.32)
    ax_t = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    # Panel A: exact data-coordinate timeline. Use [start, end+1) intervals so
    # the 2022 orange block is closed and visibly reaches the end of 2022.
    panel_label(ax_t, "A", x=-0.035, y=1.02)
    ax_t.set_xlim(args.dev_years_start - 0.5, args.test_years_end + 1.1)
    ax_t.set_ylim(0, 1)
    ax_t.set_yticks([])
    for sp in ax_t.spines.values():
        sp.set_visible(False)
    ax_t.grid(False)
    years = list(range(args.dev_years_start, args.test_years_end + 1))
    ax_t.set_xticks(years)
    ax_t.tick_params(axis="x", labelsize=8.5, length=4, pad=4)
    ybar, hbar = 0.42, 0.20
    ax_t.add_patch(Rectangle((args.dev_years_start, ybar), args.dev_years_end + 1 - args.dev_years_start, hbar,
                             facecolor="#E8F0FA", edgecolor=PUB["blue"], lw=1.0, clip_on=False, zorder=2))
    ax_t.add_patch(Rectangle((args.test_years_start, ybar), args.test_years_end + 1 - args.test_years_start, hbar,
                             facecolor="#FFF0E3", edgecolor=PUB["orange"], lw=1.0, clip_on=False, zorder=2))
    ax_t.axvline(args.test_years_start, ymin=ybar, ymax=ybar+hbar, color="#9AA6B2", lw=1.0, ls="--", zorder=3)
    ax_t.text((args.dev_years_start + args.dev_years_end + 1)/2, ybar+hbar/2, f"{args.dev_years_start}–{args.dev_years_end}",
              ha="center", va="center", fontsize=13, fontweight="bold", color=PUB["navy"], zorder=4)
    ax_t.text((args.test_years_start + args.test_years_end + 1)/2, ybar+hbar/2, f"{args.test_years_start}–{args.test_years_end}",
              ha="center", va="center", fontsize=13, fontweight="bold", color=PUB["orange"], zorder=4)
    # horizontal axis line under the bars, closed endpoint arrow outside orange block.
    ax_t.annotate("", xy=(args.test_years_end + 0.85, 0.31), xytext=(args.dev_years_start - 0.25, 0.31),
                  arrowprops=dict(arrowstyle="-|>", lw=1.2, color=PUB["navy"]), clip_on=False)
    ax_t.text(args.test_years_end + 0.98, 0.31, "Year", ha="left", va="center", fontsize=8.8, fontweight="bold")
    ax_t.text((args.dev_years_start + args.dev_years_end + 1)/2, 0.80,
              "Development cohort:\nmodel development, cross-validation,\nhyperparameter tuning, threshold locking",
              ha="center", va="center", fontsize=9.5, color=PUB["navy"], linespacing=1.20)
    ax_t.text((args.test_years_start + args.test_years_end + 1)/2, 0.80,
              "Temporal test cohort:\none-time out-of-time evaluation",
              ha="center", va="center", fontsize=9.5, color=PUB["orange"], linespacing=1.20)
    ax_t.text(0.5, 0.10, "Diagnosis year used for temporal split only; not used as predictor.",
              transform=ax_t.transAxes, ha="center", va="center", fontsize=8.7,
              bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#B8C4D0"))
    ax_t.set_title("Temporal validation schema", loc="left", fontsize=11.5, fontweight="bold", pad=8)
    add_panel_box(ax_t)

    # Panel B: equal-height cards, aligned baseline, no overlapping Excludes note.
    panel_label(ax_b, "B", x=-0.035, y=1.02)
    ax_b.axis("off")
    ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1)
    ax_b.text(0.02, 0.955, "Information-tier hierarchy", fontsize=11.5, fontweight="bold", transform=ax_b.transAxes)

    cards = [
        (0.035, 0.19, 0.285, 0.68, PUB["blue"], "Tier 1", "Earliest-available\ndemographic information",
         ["Age", "Sex", "Race/ethnicity", "Marital status"], None, "person"),
        (0.365, 0.19, 0.285, 0.68, PUB["teal"], "Tier 2", "Purity-restricted\ndiagnosis-period information",
         ["Explicit T subgroup", "Explicit N subgroup", "Explicit grade", "Observed tumor size", "Diagnostic confirmation"],
         "Excludes NOS / NX /\nunknown / missing categories", "clip"),
        (0.695, 0.19, 0.285, 0.68, PUB["orange"], "Tier 3", "Uncertainty-enriched\ndiagnosis-period information",
         ["Retains T-NOS, NX, grade unknown", "Tumor-size missing indicator", "Diagnosis-period variables"],
         "★  Main analysis tier", "network"),
    ]
    for x, y, w, h, color, title, subtitle, bullets, note, icon in cards:
        face = "#FFF7F1" if color == PUB["orange"] else "#FFFFFF"
        ax_b.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
                                      facecolor=face, edgecolor=color, lw=1.0, transform=ax_b.transAxes, zorder=3))
        ax_b.add_patch(Circle((x + 0.052, y + h - 0.095), 0.035, transform=ax_b.transAxes, facecolor=color, edgecolor=color, alpha=0.96, zorder=4))
        # Use simple glyph-like text; keep it centered so no stray bullets overlap with title text.
        glyph = "●" if icon == "person" else ("▣" if icon == "clip" else "◇")
        ax_b.text(x + 0.052, y + h - 0.095, glyph, ha="center", va="center", fontsize=15, color="white", transform=ax_b.transAxes, zorder=5)
        ax_b.text(x + 0.095, y + h - 0.065, title, ha="left", va="top", fontsize=10.4, fontweight="bold", color=color, transform=ax_b.transAxes)
        ax_b.text(x + 0.095, y + h - 0.135, subtitle, ha="left", va="top", fontsize=9.0, fontweight="bold", color=color, transform=ax_b.transAxes, linespacing=1.12)
        ax_b.plot([x + 0.025, x + w - 0.025], [y + h - 0.210, y + h - 0.210], color=color, lw=0.8, transform=ax_b.transAxes)
        by = y + h - 0.275
        for j, b in enumerate(bullets):
            yy = by - j*0.066
            ax_b.text(x + 0.048, yy, "•", fontsize=10.5, ha="left", va="center", transform=ax_b.transAxes, color=PUB["text"])
            ax_b.text(x + 0.075, yy, b, fontsize=8.7, ha="left", va="center", transform=ax_b.transAxes, color=PUB["text"], linespacing=1.10)
        if note:
            note_y = y + 0.075
            note_h = 0.102
            ax_b.add_patch(FancyBboxPatch((x + 0.035, note_y), w - 0.070, note_h,
                                          boxstyle="round,pad=0.010,rounding_size=0.012",
                                          facecolor="#F4FBFB" if color == PUB["teal"] else "#FFF2E8",
                                          edgecolor=color, lw=0.8, transform=ax_b.transAxes, zorder=4))
            ax_b.text(x + w/2, note_y + note_h/2, note, ha="center", va="center", fontsize=8.0,
                      color=color, fontweight="bold" if color == PUB["orange"] else "normal",
                      transform=ax_b.transAxes, linespacing=1.10, zorder=5)
    # Low-key arrows between cards; aligned with card centers and not touching text.
    for cx in [0.342, 0.672]:
        ax_b.text(cx, 0.535, "›", fontsize=24, color=PUB["navy"], ha="center", va="center", transform=ax_b.transAxes)
    # Ablation note sits below the cards with adequate whitespace.
    ax_b.add_patch(FancyBboxPatch((0.345, 0.045), 0.310, 0.075, boxstyle="round,pad=0.012,rounding_size=0.012",
                                  facecolor="white", edgecolor=PUB["orange"], lw=1.0, transform=ax_b.transAxes))
    ax_b.text(0.500, 0.082, "↻  Same-cohort uncertainty-masked ablation\nperformed within Tier 3",
              ha="center", va="center", fontsize=8.1, color=PUB["orange"], transform=ax_b.transAxes, linespacing=1.05)
    add_panel_box(ax_b)
    save_pub_figure(fig, outpath)
    plt.close(fig)

def save_performance_progression(table3: pd.DataFrame, outpath: Path) -> None:
    """Figure 3: performance across information tiers."""
    ok = table3.dropna(subset=["ROC-AUC", "PR-AUC"], how="all").copy()
    if ok.empty:
        return
    tier_order = ["Tier1", "Tier2", "Tier3"]
    xmap = {t: i for i, t in enumerate(tier_order)}
    models = [m for m in ["Linear Logistic", "Spline Logistic", "XGBoost", "CatBoost"] if m in ok["Model"].unique()]
    pal = get_model_palette(models)
    fig = plt.figure(figsize=(11.2, 7.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30)
    ax_auc = fig.add_subplot(gs[0, 0]); ax_pr = fig.add_subplot(gs[0, 1]); ax_f1 = fig.add_subplot(gs[1, 0]); ax_brier = fig.add_subplot(gs[1, 1])
    for ax, metric, title, higher in [(ax_auc, "ROC-AUC", "Discrimination across information tiers", True), (ax_pr, "PR-AUC", "Precision–recall performance across tiers", True), (ax_f1, "F1", "Locked-threshold F1 across tiers", True), (ax_brier, "Brier", "Calibration error across tiers", False)]:
        panel_label(ax, chr(ord('A') + [ax_auc, ax_pr, ax_f1, ax_brier].index(ax)))
        for model in models:
            sub = ok[(ok["Model"] == model) & (ok["Tier"].isin(tier_order))].copy()
            sub["x"] = sub["Tier"].map(xmap)
            sub = sub.sort_values("x")
            if sub.empty or metric not in sub:
                continue
            ax.plot(sub["x"], sub[metric], marker="o", lw=2.0, color=pal[model], label=model)
        ax.set_xticks(range(len(tier_order))); ax.set_xticklabels(["Tier 1", "Tier 2", "Tier 3"])
        ax.set_ylabel(metric)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        if metric != "Brier":
            ymax = min(1.0, max(0.35, np.nanmax(ok[metric]) + 0.08))
            ymin = max(0, np.nanmin(ok[metric]) - 0.08)
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(0, max(0.18, np.nanmax(ok[metric]) + 0.02))
        add_panel_box(ax)
    ax_auc.legend(loc="best", fontsize=8)
    save_pub_figure(fig, outpath)
    plt.close(fig)


def calibration_curve_points(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], q=min(bins, max(2, df["p"].nunique())), duplicates="drop")
    out = df.groupby("bin", observed=True).agg(mean_pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index(drop=True)
    return out



def save_tier3_head_to_head(pred_df: pd.DataFrame, table3: pd.DataFrame, outpath: Path) -> None:
    """Figure 4: Tier 3 model evaluation; no embedded figure title.

    Panel D was redesigned to prevent the selected-model callout from covering
    the CatBoost row. Model selection wording is kept in the table footnote and
    manuscript caption rather than as a large overlay inside the panel.
    """
    sub = pred_df[pred_df["Tier"] == "Tier3"].copy()
    if sub.empty:
        return
    preferred = ["Linear Logistic", "Spline Logistic", "XGBoost", "CatBoost"]
    models = [m for m in preferred if m in sub["Model"].unique()]
    pal = get_model_palette(models)
    fig = plt.figure(figsize=(13.4, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.26)
    ax_roc = fig.add_subplot(gs[0, 0]); ax_pr = fig.add_subplot(gs[0, 1]); ax_cal = fig.add_subplot(gs[1, 0]); ax_tab = fig.add_subplot(gs[1, 1])

    panel_label(ax_roc, "A"); panel_label(ax_pr, "B"); panel_label(ax_cal, "C"); panel_label(ax_tab, "D")
    for model in models:
        msub = sub[sub["Model"] == model]
        y = msub["y_true"].to_numpy(); p = msub["pred_prob"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        auc = safe_auc(y, p); ap = safe_ap(y, p)
        ax_roc.plot(fpr, tpr, lw=2.0, color=pal[model], label=f"{model} (ROC-AUC {auc:.3f})")
        ax_pr.plot(rec, prec, lw=2.0, color=pal[model], label=f"{model} (PR-AUC {ap:.3f})")
        cal = calibration_curve_points(y, p)
        ax_cal.plot(cal["mean_pred"], cal["obs"], marker="o", lw=1.8, color=pal[model], label=model)
    ax_roc.plot([0, 1], [0, 1], ls="--", color="#9AA6B2", lw=1)
    ax_roc.set(xlabel="False positive rate (1–Specificity)", ylabel="True positive rate (Sensitivity)", title="ROC curves (temporal test cohort)")
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.02); ax_roc.legend(fontsize=8, loc="lower right")

    prevalence = sub["y_true"].mean()
    ax_pr.axhline(prevalence, ls="--", color="#9AA6B2", lw=1, label=f"Prevalence ({prevalence*100:.1f}%)")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curves (temporal test cohort)")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.02); ax_pr.legend(fontsize=8, loc="upper right")

    ax_cal.plot([0, 1], [0, 1], ls="--", color="#9AA6B2", lw=1, label="Ideal")
    ax_cal.set(xlabel="Predicted probability", ylabel="Observed probability", title="Calibration curves (temporal test cohort)")
    ax_cal.set_xlim(0, min(1.0, max(0.35, sub["pred_prob"].quantile(0.995) + 0.05)))
    ax_cal.set_ylim(0, min(1.0, max(0.35, sub["y_true"].mean() * 5)))
    ax_cal.legend(fontsize=7.8, loc="upper left")
    ax_cal.text(0.98, 0.08, f"Temporal test cohort\nn={fmt_n(len(sub[sub['Model']==models[0]]))}; event rate {prevalence*100:.1f}%",
                transform=ax_cal.transAxes, ha="right", va="bottom", fontsize=8.0,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#B8C4D0"))

    # Redesigned model-summary panel.
    ax_tab.axis("off")
    ax_tab.set_title("Model performance summary (temporal test cohort)", loc="left", fontsize=11, fontweight="bold", pad=8)
    t3 = table3[table3["Tier"] == "Tier3"].copy().set_index("Model")
    rows = [m for m in models if m in t3.index]
    best_pr = t3["PR-AUC"].astype(float).idxmax() if len(t3) and "PR-AUC" in t3.columns else None
    best_roc = t3["ROC-AUC"].astype(float).idxmax() if len(t3) and "ROC-AUC" in t3.columns else None

    x_model, x_roc, x_pr, x_brier = 0.07, 0.47, 0.68, 0.88
    ax_tab.add_patch(plt.Rectangle((0.02, 0.80), 0.96, 0.14, transform=ax_tab.transAxes, facecolor=PUB["navy"], edgecolor=PUB["navy"], lw=0))
    for x, lab in [(x_model, "Model"), (x_roc, "ROC-AUC\n(Higher)"), (x_pr, "PR-AUC\n(Higher)"), (x_brier, "Brier\n(Lower)")]:
        ax_tab.text(x, 0.87, lab, color="white", fontsize=8.7, fontweight="bold", transform=ax_tab.transAxes,
                    ha="center" if x > 0.15 else "left", va="center")
    row_ys = [0.68, 0.54, 0.40, 0.26]
    for i, model in enumerate(rows[:4]):
        yrow = row_ys[i]
        is_primary = (model == best_pr)
        if is_primary:
            ax_tab.add_patch(plt.Rectangle((0.02, yrow-0.055), 0.96, 0.11, transform=ax_tab.transAxes, facecolor="#FFF3EA", edgecolor="#E6D5C7", lw=0.6))
        ax_tab.plot([0.02, 0.98], [yrow-0.075, yrow-0.075], color="#D4DCE5", lw=0.75, transform=ax_tab.transAxes)
        prefix = "★ " if is_primary else ""
        ax_tab.text(x_model, yrow, prefix + model, color=pal.get(model, PUB["text"]), fontsize=8.8,
                    fontweight="bold" if is_primary else "normal", transform=ax_tab.transAxes, ha="left", va="center")
        vals = [t3.loc[model, "ROC-AUC"], t3.loc[model, "PR-AUC"], t3.loc[model, "Brier"]]
        for x, val in [(x_roc, vals[0]), (x_pr, vals[1]), (x_brier, vals[2])]:
            ax_tab.text(x, yrow, fmt3(val), color=pal.get(model, PUB["text"]), fontsize=9.1,
                        fontweight="bold" if is_primary else "normal", transform=ax_tab.transAxes, ha="center", va="center")
    note = "★ marks highest PR-AUC (pre-specified primary selection metric)."
    if best_roc and best_pr and best_roc != best_pr:
        note += f" Highest ROC-AUC: {best_roc}."
    ax_tab.text(0.02, 0.085, note, ha="left", va="center", color=PUB["gray"], fontsize=7.8, transform=ax_tab.transAxes)
    for ax in [ax_roc, ax_pr, ax_cal, ax_tab]:
        add_panel_box(ax)
    save_pub_figure(fig, outpath)
    plt.close(fig)

def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    prevalence = y.mean()
    rows = []
    for pt in thresholds:
        pred = p >= pt
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        nb_model = tp / n - fp / n * (pt / (1 - pt))
        nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        rows.append({"threshold": pt, "model": nb_model, "treat_all": nb_all, "treat_none": 0.0})
    return pd.DataFrame(rows)




def save_risk_dca(selected: ModelResult, outpath: Path, risk_table: pd.DataFrame, args: argparse.Namespace) -> None:
    """Figure 5: selected-model risk distribution, stratification, and DCA, v2 layout."""
    y = selected.y_test
    p = selected.test_pred
    fig = plt.figure(figsize=(13.6, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.86, 1.18], width_ratios=[0.98, 1.24], hspace=0.34, wspace=0.26)
    ax_hist = fig.add_subplot(gs[0, 0]); ax_cards = fig.add_subplot(gs[0, 1]); ax_dca = fig.add_subplot(gs[1, :])
    panel_label(ax_hist, "A"); panel_label(ax_cards, "B"); panel_label(ax_dca, "C", x=-0.040)

    xmax = min(1.0, max(0.55, np.nanquantile(p, 0.995) + 0.05, selected.threshold * 1.15))
    ax_hist.hist(p, bins=40, range=(0, xmax), color=PUB["blue"], alpha=0.95, edgecolor="white", linewidth=0.25)
    ax_hist.axvline(selected.threshold, color=PUB["orange"], ls="--", lw=1.7)
    high_boundary = min(1.0, selected.threshold / max(args.risk_low_factor, 1e-6))
    if high_boundary <= xmax:
        ax_hist.axvline(high_boundary, color="#9AA6B2", ls="--", lw=1.1)
    # Keep threshold label away from the performance box.
    ytop = ax_hist.get_ylim()[1]
    tx = min(selected.threshold + 0.015, xmax*0.55)
    ax_hist.text(tx, ytop*0.87, f"Locked threshold\n{selected.threshold:.4f}", color=PUB["orange"], fontsize=8.5, fontweight="bold", ha="left", va="top")
    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("Number of patients")
    ax_hist.set_title(f"Predicted-risk distribution (temporal test cohort; {selected.model})", loc="left", fontsize=10.2, fontweight="bold")
    ax_hist.set_xlim(0, xmax)
    ax_hist.text(0.96, 0.07, f"n = {fmt_n(len(y))}", transform=ax_hist.transAxes, ha="right", va="bottom", fontsize=8.0,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#B8C4D0"))
    met = compute_metrics(y, p, selected.threshold)
    ax_hist.text(0.60, 0.88, f"Performance at locked threshold\nSensitivity  {met['Sensitivity']:.3f}\nSpecificity  {met['Specificity']:.3f}\nPPV          {met['PPV']:.3f}\nNPV          {met['NPV']:.3f}",
                 transform=ax_hist.transAxes, ha="left", va="top", fontsize=7.5,
                 bbox=dict(boxstyle="round,pad=0.26", facecolor="white", edgecolor="#B8C4D0"))

    # Panel B as compact cards with all text safely inside; no bottom overflow.
    ax_cards.axis("off")
    ax_cards.set_xlim(0, 1); ax_cards.set_ylim(0, 1)
    ax_cards.set_title("Risk-group stratification summary (temporal test cohort)", loc="left", fontsize=10.2, fontweight="bold", pad=8)
    colors = {"Low": PUB["teal"], "Intermediate": PUB["navy"], "High": PUB["orange"]}
    xs = [0.030, 0.355, 0.680]
    card_w = 0.290
    card_y = 0.075
    card_h = 0.800
    for x, (_, row) in zip(xs, risk_table.iterrows()):
        group = str(row["Risk group"])
        c = colors.get(group, PUB["blue"])
        ax_cards.add_patch(plt.Rectangle((x, card_y), card_w, card_h, transform=ax_cards.transAxes,
                                         facecolor="white", edgecolor=c, lw=1.0, zorder=2))
        ax_cards.add_patch(plt.Rectangle((x, card_y + card_h - 0.150), card_w, 0.150, transform=ax_cards.transAxes,
                                         facecolor=c, edgecolor=c, lw=0, zorder=3))
        lb = row.get("Lower probability boundary", np.nan); ub = row.get("Upper probability boundary", np.nan)
        if group == "Low": btxt = f"<{ub:.4f}"
        elif group == "High": btxt = f"≥{lb:.4f}"
        else: btxt = f"{lb:.4f}–{ub:.4f}"
        ax_cards.text(x + card_w/2, card_y + card_h - 0.075, f"{group} risk\n({btxt})", color="white", fontsize=7.4, fontweight="bold",
                      ha="center", va="center", transform=ax_cards.transAxes, linespacing=1.06, zorder=4)
        # Patient count block.
        ax_cards.text(x + card_w/2, card_y + 0.610, fmt_n(row["Number of patients"]), color=c, fontsize=15.2, fontweight="bold", ha="center", va="center", transform=ax_cards.transAxes)
        ax_cards.text(x + card_w/2, card_y + 0.552, "patients", fontsize=7.0, ha="center", va="center", transform=ax_cards.transAxes)
        # Three metric rows: label and value on same internal row to preserve height.
        metric_rows = [
            ("LM events", fmt_n(row["Number of liver metastasis events"])),
            ("Observed rate", f"{row['Observed event rate']*100:.1f}%"),
            ("Event capture", f"{row['Event capture percentage']:.1f}%"),
        ]
        ys = [card_y + 0.415, card_y + 0.280, card_y + 0.145]
        for (lab, val), yy in zip(metric_rows, ys):
            ax_cards.plot([x + 0.035, x + card_w - 0.035], [yy + 0.055, yy + 0.055], color="#D8E0E8", lw=0.7, transform=ax_cards.transAxes, zorder=3)
            ax_cards.text(x + 0.050, yy, lab, fontsize=6.8, ha="left", va="center", color=PUB["text"], transform=ax_cards.transAxes, zorder=4)
            ax_cards.text(x + card_w - 0.050, yy, val, color=c, fontsize=10.6, fontweight="bold", ha="right", va="center", transform=ax_cards.transAxes, zorder=4)

    thresholds = np.linspace(0.01, args.dca_max_threshold, 120)
    nb = net_benefit(y, p, thresholds)
    ax_dca.plot(nb["threshold"], nb["model"], lw=2.1, color=PUB["navy"], label=f"{selected.model} (selected model)")
    ax_dca.plot(nb["threshold"], nb["treat_all"], lw=1.6, color=PUB["orange"], label="Treat all")
    ax_dca.plot(nb["threshold"], nb["treat_none"], lw=1.6, color="#2C9B47", label="Treat none")
    ax_dca.axhline(0, color="#9AA6B2", lw=1, ls="--")
    ax_dca.set_xlabel("Threshold probability")
    ax_dca.set_ylabel("Net benefit")
    ax_dca.set_title("Decision-curve analysis (temporal test cohort)", loc="left", fontsize=10.2, fontweight="bold")
    ax_dca.set_xlim(0, args.dca_max_threshold)
    ymin = max(-0.10, np.nanmin(nb[["model", "treat_all", "treat_none"]].to_numpy()) - 0.02)
    ymax = np.nanmax(nb[["model", "treat_all", "treat_none"]].to_numpy()) + 0.02
    ax_dca.set_ylim(ymin, ymax)
    ax_dca.legend(fontsize=8.0, loc="upper right")
    for ax in [ax_hist, ax_cards, ax_dca]:
        add_panel_box(ax)
    save_pub_figure(fig, outpath)
    plt.close(fig)

def save_uncertainty_heatmap(profile: pd.DataFrame, outpath: Path) -> None:
    cols = ["T-NOS %", "NX %", "Grade unknown %", "Tumor size missing %"]
    if profile.empty or not set(cols).issubset(profile.columns):
        return
    mat = profile.set_index("Cohort")[cols]
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=25, ha="right")
    ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iloc[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if mat.iloc[i, j] > np.nanmax(mat.to_numpy())*0.55 else PUB["text"])
    cbar = fig.colorbar(im, ax=ax, label="Percentage")
    ax.set_title("Registry uncertainty profile across cohorts", fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    save_pub_figure(fig, outpath)
    plt.close(fig)



def save_shap_long_csv_for_postprocess(outpath: Path, feature_names: List[str], shap_values: Any, feature_values: Any) -> None:
    """Save SHAP values in a long CSV used by the integrated top-journal postprocessor.

    Expected output:
      <outdir>/data/SHAP_values_long_selected_model.csv
    Columns:
      feature, shap_value, feature_value

    This prevents the minimal SHAP figure from silently disappearing when the
    postprocessor cannot reconstruct SHAP values from a PDF.
    """
    try:
        # shap may return a list for binary classifiers; use positive-class values.
        vals = shap_values
        if isinstance(vals, list):
            vals = vals[-1]
        vals = np.asarray(vals)
        if vals.ndim == 3:
            vals = vals[:, :, -1]
        if hasattr(feature_values, "toarray"):
            Xv = feature_values.toarray()
        else:
            Xv = np.asarray(feature_values)
        if Xv.ndim == 1:
            Xv = Xv.reshape(-1, 1)
        n, m = vals.shape[0], vals.shape[1]
        names = list(feature_names)[:m]
        if len(names) < m:
            names += [f"feature_{i}" for i in range(len(names), m)]
        # Guard if feature values have a different column count after transform.
        if Xv.shape[1] != m:
            Xv = np.full((n, m), np.nan)
        rows = []
        max_rows = min(n, 5000)  # enough for beeswarm while keeping file reasonable
        for j, name in enumerate(names):
            rows.append(pd.DataFrame({
                "feature": name,
                "shap_value": vals[:max_rows, j],
                "feature_value": Xv[:max_rows, j]
            }))
        long_df = pd.concat(rows, ignore_index=True)
        outdir = outpath.parents[1] if len(outpath.parents) >= 2 else outpath.parent
        data_dir = ensure_dir(outdir / "data")
        long_df.to_csv(data_dir / "SHAP_values_long_selected_model.csv", index=False, encoding="utf-8-sig")
        # Also write a compact importance table, useful for review and fallback figures.
        imp = long_df.groupby("feature")["shap_value"].apply(lambda x: float(np.nanmean(np.abs(x)))).reset_index()
        imp = imp.rename(columns={"shap_value": "mean_abs_shap"}).sort_values("mean_abs_shap", ascending=False)
        imp.to_csv(data_dir / "SHAP_grouped_importance_selected_model.csv", index=False, encoding="utf-8-sig")
        logging.info("Saved SHAP long CSV for topjournal postprocess: %s", data_dir / "SHAP_values_long_selected_model.csv")
    except Exception:
        logging.exception("Failed to save SHAP long CSV for postprocess.")


def save_shap_or_importance(selected: ModelResult, outpath: Path, args: argparse.Namespace) -> None:
    if args.skip_shap:
        return
    try:
        if selected.model == "XGBoost" and HAS_SHAP:
            pipe = selected.estimator
            pre = pipe.named_steps["preprocess"]
            clf = pipe.named_steps["clf"]
            X_trans = pre.transform(selected.x_test)
            names = pre.get_feature_names_out()
            explainer = shap.TreeExplainer(clf)
            values = explainer.shap_values(X_trans)
            save_shap_long_csv_for_postprocess(outpath, list(names), values, X_trans)
            fig = plt.figure(figsize=(8.5, 6.2))
            shap.summary_plot(values, X_trans, feature_names=names, show=False, max_display=15)
            plt.title("SHAP summary for selected model", loc="left", fontsize=12, fontweight="bold")
            fig = plt.gcf()
            fig.tight_layout()
            save_pub_figure(fig, outpath)
            plt.close(fig)
            return
        if selected.model == "CatBoost" and HAS_SHAP:
            params = selected.selected_params
            med = params.get("_cat_medians") if isinstance(params, dict) else None
            X_cb, _, _ = prepare_catboost_frame(selected.x_test, selected.feature_cols, selected.numeric_cols, selected.categorical_cols, med)
            explainer = shap.TreeExplainer(selected.estimator)
            values = explainer.shap_values(X_cb)
            save_shap_long_csv_for_postprocess(outpath, selected.feature_cols, values, X_cb)
            fig = plt.figure(figsize=(8.5, 6.2))
            shap.summary_plot(values, X_cb, feature_names=selected.feature_cols, show=False, max_display=15)
            plt.title("SHAP summary for selected model", loc="left", fontsize=12, fontweight="bold")
            fig = plt.gcf(); fig.tight_layout()
            save_pub_figure(fig, outpath)
            plt.close(fig)
            return
    except Exception:
        logging.exception("SHAP failed; falling back to permutation importance.")

    try:
        def scorer(est, X, y):
            p = predict_proba_model(est, X, selected.model, selected.feature_cols, selected.numeric_cols, selected.categorical_cols,
                                    selected.selected_params.get("_cat_medians") if isinstance(selected.selected_params, dict) else None)
            return average_precision_score(y, p)
        perm = permutation_importance(selected.estimator, selected.x_test, selected.y_test, scoring=scorer, n_repeats=10, random_state=args.random_state)
        imp = pd.DataFrame({"feature": selected.feature_cols, "importance": perm.importances_mean}).sort_values("importance", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8.0, 5.6))
        panel_label(ax, "A")
        ax.barh(imp["feature"][::-1], imp["importance"][::-1], color=PUB["blue"])
        ax.set_xlabel("Permutation importance, mean decrease in average precision")
        ax.set_title("Fallback model importance", loc="left", fontsize=12, fontweight="bold")
        fig.tight_layout()
        save_pub_figure(fig, outpath)
        plt.close(fig)
    except Exception:
        logging.exception("Fallback importance failed.")


def save_simple_pdp(selected: ModelResult, outpath: Path) -> None:
    features = []
    if "Age_Years" in selected.feature_cols:
        features.append("Age_Years")
    if "Tumor_Size_Model" in selected.feature_cols:
        features.append("Tumor_Size_Model")
    if not features:
        return
    fig, axes = plt.subplots(1, len(features), figsize=(5.4 * len(features), 4.2))
    if len(features) == 1:
        axes = [axes]
    for i, (ax, feat) in enumerate(zip(axes, features)):
        panel_label(ax, chr(ord('A') + i))
        X0 = selected.x_test.copy()
        vals = pd.to_numeric(X0[feat], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        lo, hi = vals.quantile(0.05), vals.quantile(0.95)
        grid = np.linspace(lo, hi, 24)
        means = []
        for v in grid:
            Xg = X0.copy(); Xg[feat] = v
            med = selected.selected_params.get("_cat_medians") if isinstance(selected.selected_params, dict) else None
            pp = predict_proba_model(selected.estimator, Xg, selected.model, selected.feature_cols, selected.numeric_cols, selected.categorical_cols, med)
            means.append(np.mean(pp))
        ax.plot(grid, means, marker="o", lw=2.0, color=PUB["orange"])
        ax.fill_between(grid, means, np.minimum(means, means), color=PUB["orange"], alpha=0.1)
        label = "Age, years" if feat == "Age_Years" else "Tumor size, mm"
        ax.set_xlabel(label)
        ax.set_ylabel("Mean predicted probability")
        ax.set_title(f"Partial-dependence profile: {label}", loc="left", fontsize=11, fontweight="bold")
        add_panel_box(ax)
    fig.tight_layout()
    save_pub_figure(fig, outpath)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Consistency checks and output
# -----------------------------------------------------------------------------

def final_consistency_checks(selected: Optional[ModelResult], table3: pd.DataFrame, pred_df: pd.DataFrame, table_s7: pd.DataFrame, table4: Optional[pd.DataFrame], table_s1: Optional[pd.DataFrame]) -> List[str]:
    notes = []
    if selected is None:
        notes.append("No selected model available; some final outputs were skipped.")
        return notes

    row = table3[(table3["Tier"] == selected.tier) & (table3["Model"] == selected.model)]
    if len(row) != 1:
        raise ValueError("Selected model row not uniquely found in Table 3.")
    t3_threshold = float(row.iloc[0]["Locked threshold"])
    if abs(t3_threshold - selected.threshold) > 1e-12:
        raise ValueError("Selected model threshold differs between Table 3 and selected result object.")

    if table_s1 is not None and not table_s1.empty:
        if abs(float(table_s1.iloc[0]["Locked threshold"]) - selected.threshold) > 1e-12:
            raise ValueError("Table S1 threshold differs from selected model threshold.")

    if pred_df.empty:
        raise ValueError("Prediction dataframe is empty.")

    # Detect suspicious exact duplicate predictions for same model across tiers.
    for model in pred_df["Model"].unique():
        subs = []
        for tier in ["Tier1", "Tier2", "Tier3"]:
            s = pred_df[(pred_df["Model"] == model) & (pred_df["Tier"] == tier)].sort_values("Patient_ID")["pred_prob"].to_numpy()
            if len(s):
                subs.append((tier, s))
        for i in range(len(subs)):
            for j in range(i + 1, len(subs)):
                if len(subs[i][1]) == len(subs[j][1]) and np.allclose(subs[i][1], subs[j][1], rtol=0, atol=0):
                    notes.append(f"WARNING: {model} predictions are exactly identical for {subs[i][0]} and {subs[j][0]}; inspect tier filtering.")

    return notes




def style_excel_workbook(path: Path) -> None:
    """Apply a restrained journal-style table design to xlsx outputs.

    v2 improves visual hierarchy for baseline tables: section rows are bold with
    pale-blue fill; indented category rows remain readable. This makes the Excel
    file closer to a manuscript table and easier to edit by hand.
    """
    try:
        from openpyxl import load_workbook
        from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
        from openpyxl.utils import get_column_letter
        wb = load_workbook(path)
        header_fill = PatternFill("solid", fgColor="EAF2FB")
        section_fill = PatternFill("solid", fgColor="F2F7FC")
        alt_fill = PatternFill("solid", fgColor="FAFCFE")
        selected_fill = PatternFill("solid", fgColor="FFF3EA")
        navy = "073763"; orange = "F26B21"; gray = "6B7280"
        thin = Side(style="thin", color="B8C4D0")
        medium = Side(style="medium", color=navy)
        for ws in wb.worksheets:
            max_row, max_col = ws.max_row, ws.max_column
            for cell in ws[1]:
                cell.font = Font(bold=True, color=navy, size=10)
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                cell.border = Border(top=medium, bottom=medium)
            for r in range(2, max_row + 1):
                first = ws.cell(r, 1).value
                first_s = "" if first is None else str(first)
                row_values = [ws.cell(r, c).value for c in range(1, max_col + 1)]
                is_section = bool(first_s) and not first_s.startswith("    ") and all((ws.cell(r, c).value in (None, "")) for c in range(2, max_col + 1))
                is_selected = any(str(v).lower() == "xgboost" or "selected main model" in str(v).lower() for v in row_values if v is not None)
                for c in range(1, max_col + 1):
                    cell = ws.cell(r, c)
                    if is_section:
                        cell.fill = section_fill
                        cell.font = Font(bold=True, color=navy, size=9.5)
                        cell.border = Border(top=thin, bottom=thin)
                    elif is_selected:
                        cell.fill = selected_fill
                        cell.font = Font(color=orange, bold=True if c == 1 else False, size=9.5)
                        cell.border = Border(bottom=thin)
                    else:
                        if r % 2 == 0:
                            cell.fill = alt_fill
                        # Indented category rows get gray label, normal data.
                        if c == 1 and first_s.startswith("    "):
                            cell.font = Font(color=gray, size=9)
                        else:
                            cell.font = Font(color="000000", size=9)
                        cell.border = Border(bottom=thin)
                    align_h = "left" if c == 1 else "center"
                    cell.alignment = Alignment(horizontal=align_h, vertical="center", wrap_text=True)
            ws.freeze_panes = "A2"
            ws.sheet_view.showGridLines = False
            for c in range(1, max_col + 1):
                col = get_column_letter(c)
                vals = [ws.cell(r, c).value for r in range(1, min(max_row, 120) + 1)]
                width = max([len(str(v)) if v is not None else 0 for v in vals] + [8]) + 2
                # Baseline and mapping tables need more text width but not excessive columns.
                if c == 1:
                    width = min(max(width, 18), 44)
                else:
                    width = min(max(width, 12), 38)
                ws.column_dimensions[col].width = width
            ws.row_dimensions[1].height = 28
            for r in range(2, max_row + 1):
                first_s = "" if ws.cell(r,1).value is None else str(ws.cell(r,1).value)
                ws.row_dimensions[r].height = 24 if len(first_s) > 35 else 21
        wb.save(path)
    except Exception:
        logging.exception("Excel styling failed for %s; unstyled workbook was retained.", path)

def write_excel_outputs(outdir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    tables_dir = ensure_dir(outdir / "tables")
    for name, df in tables.items():
        if df is None or len(df) == 0:
            continue
        path = tables_dir / f"{name}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=name[:31])
        style_excel_workbook(path)

def save_csv_outputs(outdir: Path, datasets: Dict[str, pd.DataFrame], pred_df: pd.DataFrame) -> None:
    data_dir = ensure_dir(outdir / "data")
    for name, df in datasets.items():
        df.to_csv(data_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    if pred_df is not None and not pred_df.empty:
        pred_df.to_csv(data_dir / "test_set_predictions.csv", index=False, encoding="utf-8-sig")


def setup_logging(outdir: Path) -> None:
    ensure_dir(outdir)
    log_path = outdir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Corrected SEER GBC liver-metastasis risk stratification pipeline")
    p.add_argument("--workdir", default=".", help="Working directory containing input CSV")
    p.add_argument("--input", default=None, help="Raw SEER CSV file. Defaults to newgla_reextract*.csv in workdir")
    p.add_argument("--outdir", default="final_submission_suite_fixed", help="Output directory")
    p.add_argument("--dev-years-start", type=int, default=2010)
    p.add_argument("--dev-years-end", type=int, default=2018)
    p.add_argument("--test-years-start", type=int, default=2019)
    p.add_argument("--test-years-end", type=int, default=2022)
    p.add_argument("--exclude-years", default="", help="Comma-separated years to exclude, e.g. 2016,2017. Default keeps them.")
    p.add_argument("--allow-missing-2016-2017", action="store_true", help="Warn rather than fail if 2016-2017 disappear after harmonization")
    p.add_argument("--fine-t2-main", action="store_true", help="Do not collapse T2a/T2b/T2NOS to T2 in main analysis")
    p.add_argument("--treat-zero-size-missing", action="store_true", help="Treat tumor size code 000 as missing")
    p.add_argument("--adenocarcinoma-only", action="store_true", help="Restrict to broad adenocarcinoma-family histology codes")

    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--search-iters-logistic", type=int, default=10)
    p.add_argument("--search-iters-xgb", type=int, default=30)
    p.add_argument("--search-iters-cat", type=int, default=30)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--n-jobs-search", type=int, default=1)
    p.add_argument("--n-jobs-model", type=int, default=1)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--search-scoring", default="average_precision", choices=["average_precision", "roc_auc"])
    p.add_argument("--threshold-rule", default="sensitivity80", choices=["sensitivity80", "youden"])
    p.add_argument("--target-sensitivity", type=float, default=0.80)
    p.add_argument("--selection-metric", default="PR-AUC", choices=["PR-AUC", "ROC-AUC", "F1"])
    p.add_argument("--selection-strategy", default="balanced", choices=["primary_metric", "balanced"], help="How to select the final Tier 3 model after temporal testing. 'balanced' penalizes poor Brier score.")
    p.add_argument("--tree-objective", default="balanced", choices=["balanced", "pr_auc", "roc_auc", "brier", "f1"], help="OOF objective for XGBoost/CatBoost hyperparameter tuning.")
    p.add_argument("--tree-calibration", default="both", choices=["none", "sigmoid", "isotonic", "both"], help="Development-only calibration for tree model probabilities. 'both' chooses sigmoid/isotonic/none by OOF balanced score.")
    p.add_argument("--cal-slope-min", type=float, default=0.80)
    p.add_argument("--cal-slope-max", type=float, default=1.25)
    p.add_argument("--risk-low-factor", type=float, default=0.5)
    p.add_argument("--dca-max-threshold", type=float, default=0.50)
    p.add_argument("--xgb-tree-method", default="hist")

    p.add_argument("--skip-xgboost", action="store_true")
    p.add_argument("--skip-catboost", action="store_true")
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("--fast", action="store_true", help="Debug mode: small searches and small bootstrap")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    workdir = Path(args.workdir).resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = workdir / outdir
    ensure_dir(outdir)
    setup_logging(outdir)
    set_publication_style()

    if args.fast:
        args.bootstrap = min(args.bootstrap, 50)
        args.cv_folds = min(args.cv_folds, 3)
        args.search_iters_logistic = min(args.search_iters_logistic, 2)
        args.search_iters_xgb = min(args.search_iters_xgb, 2)
        args.search_iters_cat = min(args.search_iters_cat, 2)
        logging.info("FAST mode enabled")

    try:
        input_path = infer_input_file(workdir, args.input)
        logging.info("Input file: %s", input_path)
        raw = pd.read_csv(input_path, low_memory=False)
        logging.info("Raw shape: %s", raw.shape)

        main_df, flow = build_main_cohort(raw, args)
        tiers = make_tier_datasets(main_df)
        preflight_notes = preflight_checks(raw, main_df, tiers, args)
        for note in preflight_notes:
            logging.warning(note) if note.startswith("WARNING") else logging.info(note)

        datasets = {
            "01_main_cohort_cleaned_fixed": main_df,
            "01_model_ready_Tier1_fixed": tiers["Tier1"],
            "01_model_ready_Tier2_fixed": tiers["Tier2"],
            "01_model_ready_Tier3_fixed": tiers["Tier3"],
        }

        results, pred_df, table3, table_s4 = run_all_models(tiers, args)
        table_s7 = make_boosted_tree_head_to_head(pred_df, table3)
        selected = select_main_model(results, table3, args)

        table4 = pd.DataFrame()
        table_s1 = pd.DataFrame()
        table_s2 = pd.DataFrame()
        selected_summary: Dict[str, Any] = {}
        if selected is not None:
            table4 = make_risk_table(selected.y_test, selected.test_pred, selected.threshold, low_factor=args.risk_low_factor)
            table_s1 = make_operating_characteristics(selected)
            table_s2 = run_same_cohort_ablation(tiers["Tier3"], selected.model, args)
            selected_summary = {
                "selected_tier": selected.tier,
                "selected_model": selected.model,
                "locked_threshold": selected.threshold,
                "threshold_rule": selected.threshold_rule,
                "selection_metric": args.selection_metric,
                "feature_columns": selected.feature_cols,
                "selected_params": sanitize_json(selected.selected_params),
            }
            joblib.dump(selected.estimator, outdir / "selected_model.pkl")

        table1 = make_baseline_table(main_df, group_col="Temporal_Split")
        table2 = predictor_timing_table()
        table_s3 = logistic_coefficients(results)
        table_s5 = make_baseline_table(main_df, group_col=TARGET_COL)
        table_s6 = uncertainty_profile(main_df)
        table_s8 = mapping_table()

        final_notes = final_consistency_checks(selected, table3, pred_df, table_s7, table4, table_s1)
        all_notes = preflight_notes + final_notes

        tables = {
            "Table1_Baseline_Overall_Development_TemporalTest": table1,
            "Table2_Predictor_Timing_and_Information_Hierarchy": table2,
            "Table3_Model_Performance": table3,
            "Table4_LockedThreshold_Risk_Stratification": table4,
            "TableS1_Operating_Characteristics": table_s1,
            "TableS2_SameCohort_Ablation": table_s2,
            "TableS3_LogisticBaseline_Coefficients": table_s3,
            "TableS4_Hyperparameter_Grid_and_Selected_Params": table_s4,
            "TableS5_Baseline_by_Liver_Metastasis_Status": table_s5,
            "TableS6_Development_vs_TemporalTest_Uncertainty_Profile": table_s6,
            "TableS7_BoostedTree_HeadToHead": table_s7,
            "TableS8_Source_to_Harmonized_Variable_Mapping": table_s8,
        }

        write_excel_outputs(outdir, tables)
        save_csv_outputs(outdir, datasets, pred_df)

        figures_dir = ensure_dir(outdir / "figures")
        save_flowchart(flow, tiers, figures_dir / "Figure1_Study_Flowchart.pdf")
        save_schema_figure(args, figures_dir / "Figure2_TemporalValidation_InformationHierarchy.pdf")
        save_performance_progression(table3, figures_dir / "Figure3_Performance_Progression_Across_Tiers.pdf")
        save_tier3_head_to_head(pred_df, table3, figures_dir / "Figure4_Tier3_Model_HeadToHead.pdf")
        if selected is not None:
            save_risk_dca(selected, figures_dir / "Figure5_LockedThreshold_RiskStratification_DCA.pdf", table4, args)
            save_shap_or_importance(selected, figures_dir / "FigureS1_SHAP_or_ModelImportance.pdf", args)
            save_simple_pdp(selected, figures_dir / "FigureS2_TruePDP_SelectedModel.pdf")
        save_uncertainty_heatmap(table_s6, figures_dir / "FigureS3_UncertaintyStructure_Heatmap.pdf")

        with open(outdir / "cohort_flow_counts.json", "w", encoding="utf-8") as f:
            json.dump(flow, f, indent=2, ensure_ascii=False)
        with open(outdir / "selected_model_summary.json", "w", encoding="utf-8") as f:
            json.dump(selected_summary, f, indent=2, ensure_ascii=False)

        manifest = {
            "script": Path(__file__).name,
            "input_file": str(input_path),
            "outdir": str(outdir),
            "arguments": vars(args),
            "packages": {
                "xgboost_available": HAS_XGB,
                "catboost_available": HAS_CAT,
                "shap_available": HAS_SHAP,
                "statsmodels_available": HAS_SM,
            },
            "flow": flow,
            "tier_sizes": {k: {"n": int(len(v)), "events": int(v[TARGET_COL].sum())} for k, v in tiers.items()},
            "selected_model_summary": selected_summary,
            "notes": all_notes,
        }
        with open(outdir / "final_run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        with open(outdir / "final_run_manifest.txt", "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest, indent=2, ensure_ascii=False))

        logging.info("Completed successfully. Outputs written to: %s", outdir)
        if all_notes:
            logging.info("Notes/warnings:\n%s", "\n".join(all_notes))

    except Exception:
        err = traceback.format_exc()
        with open(outdir / "run_error.log", "w", encoding="utf-8") as f:
            f.write(err)
        logging.error("Pipeline failed. Full traceback written to run_error.log")
        raise


if __name__ == "__main__":
    main()
