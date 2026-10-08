# Round 2 minor-revision clarification: registry-input recoding and redevelopment sensitivity analysis

This note documents the implemented procedure for the same-cohort registry-input recoding and redevelopment sensitivity analysis.
It is a clarification of the archived workflow; the reported performance estimates are unchanged.

## What was recoded

- `Unified_T`: T1a, T1b, and T1-NOS were collapsed to T1; T2 categories were collapsed to T2.
- `Unified_N`: NX was recoded to N0.
- `Unified_Grade`: Unknown was recoded to G2.
- `Missing_Size_Indicator`: set to 0.
- The imputed tumor-size value itself was retained.

## Development and evaluation procedure

The archived function `run_same_cohort_ablation` calls `run_one_model` separately for the full Tier 3 branch and the recoded branch. Therefore the recoded branch was not a test-time perturbation of an unchanged estimator. Hyperparameter search, final fitting, development-only probability-calibration selection/fitting, and threshold selection were repeated using the same development cohort and the same rules.

The threshold-selection rule was the same in both branches, but the numerical thresholds differed:

- Full Tier 3 CatBoost: `0.08367640386406207`
- Recoded/redeveloped Tier 3 CatBoost: `0.1196172248803828`

Threshold-dependent metrics in the current manuscript and Supplementary Table S2 are interpreted at each branch's own development-derived threshold.

## Interpretation boundary

Because the recoding also reduces T-subcategory detail and the model is redeveloped, the observed performance attenuation reflects the combined recoding-and-redevelopment procedure. It should not be interpreted as the isolated contribution of registry-uncertainty indicators.
