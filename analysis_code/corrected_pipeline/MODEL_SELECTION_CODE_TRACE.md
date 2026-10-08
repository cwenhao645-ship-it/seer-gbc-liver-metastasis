# Model designation code trace and post hoc verification

## Severity

High reporting and reproducibility risk; no demonstrated change to the fitted estimator, calibration, or locked threshold.

## Finding

The archived workflow constructed its Tier 3 reporting table from temporal-test predictions and passed that table to `select_main_model`. Under the recorded `balanced` strategy, the reporting function sorted a composite containing temporal-test metrics when choosing which fixed model to emphasize. This is inconsistent with the prior manuscript statement that CatBoost had been prospectively designated using development data alone.

The candidate estimators, hyperparameters, calibration procedures, and model-specific thresholds were nevertheless fitted or derived in the 2010 to 2018 development period. The defect concerns the report-level choice of which fixed model to emphasize and its description.

## Independent post hoc audit

The audit fixed every recorded Tier 3 estimator and hyperparameter, reconstructed five-fold out-of-fold predictions within the 2010 to 2018 development cohort, and reapplied the documented balanced composite without accessing the 2019 to 2022 temporal-test outcomes.

| Model | Development OOF composite | ROC-AUC | PR-AUC | Brier score | F1 | Calibration slope |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 1.243304 | 0.765650 | 0.338232 | 0.092870 | 0.311780 | 1.002883 |
| XGBoost | 1.242967 | 0.763377 | 0.338270 | 0.093146 | 0.316547 | 1.003097 |
| Spline logistic regression | 1.222995 | 0.763671 | 0.337807 | 0.093103 | 0.310680 | 1.118213 |
| Linear logistic regression | 1.215618 | 0.762304 | 0.337225 | 0.093209 | 0.302796 | 1.133514 |

CatBoost ranked first by 0.000337 over XGBoost. The difference is too small to establish a stable algorithmic hierarchy. The reconstructed CatBoost threshold was 0.08367640386406207, exactly equal to the archived locked threshold, with absolute difference 0.0.

## Required correction

1. Do not state that the archived report-level designation function was fully independent of temporal-test results.
2. State that a post hoc revision audit identified the code-path inconsistency, ranked CatBoost marginally first, and reproduced its threshold exactly.
3. Do not refit, retune, recalibrate, or replace the model after seeing temporal-test results.
4. Preserve the archived script as provenance. Supply a corrected designation implementation and a machine-readable audit result in the reproducibility package.
5. Treat all four fixed Tier 3 candidates as the primary temporal comparison and use CatBoost only as a representative model for threshold and enrichment displays.
6. Retain the limitation that this verification was performed after temporal-test results were known and therefore does not recreate prospectively locked model selection.

## Evidence

- Audit script: `analysis_code/reviewer1_revision/model_selection_audit.py`
- Aggregate output: `aggregate_outputs/development_oof_model_selection_audit.csv`
- Machine-readable result: `aggregate_outputs/model_selection_audit.json`
