# Reproduction routes

## 1. Inspect supplied results without private inputs

`aggregate_outputs/` and `model_metadata/` contain the distributed estimates,
bootstrap/simulation summary statistics, tuning records, and variable mappings.
You can inspect these without access to individual SEER records.

`python verify_release.py` checks file integrity and structure, not numerical
reproduction of the study. All supplied analysis scripts were syntax-checked
when packaged; the full study was not rerun for this release.

## 2. Rerun the historical pipeline

Use Python 3.13 and the recorded dependency versions in `requirements.txt`.
The unpinned `Pillow` entry was already unpinned in the source environment;
`joblib` is imported by the scripts and is normally installed through
scikit-learn. A frozen record for every transitive dependency is not available.
This is a recorded research environment, not a claim of a fully locked container.

Place an authorized copy of the same SEER extract outside this repository.
From the repository root:

```console
python run_analysis.py --input ../private_seer/seer_extract.csv --output-dir ../private_seer_run --dry-run
python run_analysis.py --input ../private_seer/seer_extract.csv --output-dir ../private_seer_run
```

The helper forwards the original run-manifest settings, including the original
search budgets, rather than relying on the scripts' smaller default tree-search
budgets. It does not modify the historical script. The output must be a new or
empty private directory. The historical pipeline has the documented report-level
designation issue; preserve that distinction when interpreting its output.

Adding `--pipeline corrected` runs the development-only designation reference
instead. It does not change the provenance of the reported historical analyses.
An actual rerun can differ across inputs, platforms, libraries, or thread
implementations; exact agreement should be checked against the supplied outputs,
not assumed from successful execution.

## 3. Development-OOF designation audit

With an authorized extract, this audit uses the archived 12-record parameter
workbook and reconstructs development-OOF predictions for the four fixed Tier 3
candidates. It does not require a publicly supplied fitted model:

```console
python analysis_code/reviewer1_revision/model_selection_audit.py --pipeline analysis_code/corrected_pipeline/_embedded_modelopt_calibrated_DEVELOPMENT_OOF_CORRECTED.py --raw ../private_seer/seer_extract.csv --table-s4 aggregate_outputs/TableS4_Hyperparameter_Grid_and_Selected_Params.xlsx --outdir ../private_seer_audit
```

The audit remains post hoc: temporal-test results had already been known when it
was conducted. Its supplied result is not prospective model selection.

## 4. Outcome-status, paired, and subgroup analyses

These scripts need locally held locked artifacts. They are deliberately absent
from the public repository. A pipeline rerun generates the expected layout:

```text
../private_seer_run/
  selected_model.pkl
  selected_model_summary.json
  data/
    01_model_ready_Tier3_fixed.csv
    test_set_predictions.csv
    SHAP_values_long_selected_model.csv
```

The SHAP file is present only if SHAP calculation was completed. Exact historical
replication requires the authors' original authorized locked artifacts or a
verified reproduction of them. Newly regenerated artifacts must not silently be
described as the original locked artifacts.

```console
python analysis_code/reviewer1_revision/run_reviewer1_analyses.py --raw ../private_seer/seer_extract.csv --locked-dir ../private_seer_run --original-script analysis_code/archived_pipeline/_embedded_modelopt_calibrated_ARCHIVED.py --outdir ../private_seer_revision --tables-dir ../private_seer_revision_tables
python analysis_code/reviewer1_revision/run_reporting_subgroup_analysis.py --predictions ../private_seer_run/data/test_set_predictions.csv --model-ready ../private_seer_run/data/01_model_ready_Tier3_fixed.csv --output-dir ../private_seer_subgroups
python analysis_code/reviewer1_revision/redraw_final_figures.py --predictions ../private_seer_run/data/test_set_predictions.csv --shap ../private_seer_run/data/SHAP_values_long_selected_model.csv --figures-dir ../private_seer_figures --source-dir ../private_seer_figure_sources
```

Generated tables and figure-source files need a fresh disclosure review before
publication. In particular, the original plotting script writes case-level
SHAP source rows; that output is not the aggregated SHAP file distributed here.
Figure 5 in this repository is the final revision figure; the historical
plotting scripts are not asserted to regenerate every final revision label.

## 5. Recoding and redevelopment

Both main pipelines expose `run_same_cohort_ablation`. They run model development
separately for the full and recoded conditions. These threshold-dependent results
must be interpreted at their respective development-derived thresholds, not a
single fixed threshold. The clarification CSV and method note record the
historical result; do not relabel it as fixed-model test-time masking.
