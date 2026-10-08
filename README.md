# SEER gallbladder cancer: temporal validation of liver-metastasis risk enrichment

Code and aggregate research materials accompanying:

**Temporal validation of liver metastasis risk enrichment at diagnosis in T1 and T2 gallbladder cancer using SEER registry data**

This is the 8 October 2026 minor-revision snapshot, prepared for public archiving.
The manuscript is under review. The public repository is
[cwenhao645-ship-it/seer-gbc-liver-metastasis](https://github.com/cwenhao645-ship-it/seer-gbc-liver-metastasis).
No article DOI or archive DOI has been assigned at the time of this snapshot.

## Scope

The study compares four fixed algorithms across three information tiers, with
development in 2010-2018 and temporal evaluation in 2019-2022. All fixed Tier 3
candidates are the primary comparison. CatBoost is a representative model for
exploratory threshold, enrichment, sensitivity, and SHAP analyses; it is not a
bedside diagnostic tool or a model intended for clinical deployment.

The registry-input recoding and redevelopment analysis repeated model
development, calibration, and threshold selection. It was **not** evaluation-only
masking of an unchanged estimator. The full and recoded branches used their own
development-derived thresholds: `0.08367640386406207` and `0.1196172248803828`.
See [the method clarification](ROUND2_METHOD_CLARIFICATION.md).

## Files

| Location | Contents |
| --- | --- |
| `analysis_code/archived_pipeline/` | Unchanged historical analysis pipeline. |
| `analysis_code/corrected_pipeline/` | Development-OOF-only model-designation reference for future reruns, with the original diff and a code-trace note. |
| `analysis_code/reviewer1_revision/` | Outcome-observation, IPW, pattern-mixture, paired-comparison, subgroup, audit, and figure scripts. Historical filenames are retained. |
| `aggregate_outputs/` | Aggregate results and bootstrap/simulation summary statistics, not patient records. |
| `aggregate_outputs/figure_source/` | ROC/PR coordinates without case-level scores, calibration/DCA/risk summaries, and aggregated SHAP summaries. |
| `model_metadata/` | Run settings, software versions, all 12 candidate tuning records, representative-model metadata, and the 21-variable data dictionary. No fitted estimator is included. |
| `figures/` | Final manuscript Figures 1, 3, 4, and 5, copied without alteration. Case-point SHAP graphics are not included in this public snapshot. |
| `provenance/` | Checksums, file inventory, and the mapping from the peer-review ZIP to the public snapshot. |
| `docs/` | Reproduction, data access, disclosure handling, licensing, and publication instructions. |

The original selected-parameter workbook is preserved at
`aggregate_outputs/TableS4_Hyperparameter_Grid_and_Selected_Params.xlsx`.
Its 12 records are also provided as CSV and JSON in `model_metadata/`.
These are tuning settings and calibration-method labels, not fitted tree weights
or a distributable clinical model.

## Check this download without SEER access

Python 3.13 was used in the archived environment. The integrity check needs only
the Python standard library:

```console
python verify_release.py
```

It checks every listed checksum, missing/extra files, Python syntax, all 12 tuning
records, the 21-variable dictionary, and basic restricted-file/header screening.
It does not rerun the statistical analyses or certify legal compliance.

## Reproduce with an authorized extract

Individual-level SEER data are not supplied. You need lawful access to the same
SEER Research Plus extract: 17 Registries, November 2024 submission, 2000-2022.
Another release or the Research-only product is not assumed to be equivalent.
Current access eligibility is governed by SEER, not by this repository.

Install the recorded dependencies in a separate virtual environment:

```console
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate `.venv` before running the second command. Store the source extract and
all generated outputs **outside this repository**, including outside any synced
public folder. The helper below requires an explicit source file and output
directory; its default is the historical pipeline with the archived run settings,
including 120 search iterations for each tree algorithm:

```console
python run_analysis.py --input ../private_seer/seer_extract.csv --output-dir ../private_seer_run --dry-run
python run_analysis.py --input ../private_seer/seer_extract.csv --output-dir ../private_seer_run
```

For the future development-only designation reference, add
`--pipeline corrected`. This reference must not be represented as the historical
workflow or as evidence that the original designation was prospective.
See [REPRODUCING.md](docs/REPRODUCING.md) for downstream commands and limitations.

## Historical model-designation boundary

The historical reporting function used temporal-test summaries to choose which
fixed model to emphasize. The estimators, calibration procedures, and individual
thresholds were developed in the development period. A subsequent fixed-candidate
development-OOF audit ranked CatBoost only marginally above XGBoost and reproduced
the representative threshold. This post hoc check does not recreate prospective
selection and did not change the fitted models or the reported main results.

## Public disclosure handling

This repository is a public-release subset of the peer-review package, not an
identical copy. It excludes SEER case listings, identifiers, case-level
predictions/SHAP files, fitted models, confidential correspondence, and manuscripts.
Some demographic count blocks and threshold-dependent subgroup columns are
excluded from the public copies to avoid releasing or reconstructing small
demographic counts. Remaining values are unchanged; no analysis was rerun.
See [PUBLIC_RELEASE_NOTES.md](docs/PUBLIC_RELEASE_NOTES.md).

## License and citation

The authors authorized the **MIT license for code and associated software
documentation**. See [LICENSE](LICENSE). This does not license SEER data, third-party
software, aggregate datasets, or manuscript figures. See
[LICENSING.md](docs/LICENSING.md) for the scope of rights.

Software citation metadata are provided in [CITATION.cff](CITATION.cff). The two
listed software contributors are Wenhao Chen and Zhan Wang; all manuscript
authors and their agreed order are recorded in that file's description. No
publication or repository identifier is invented.

After upload, create a versioned release and archive it with Zenodo. Only then
add the real repository URL and release DOI to the manuscript and citation
metadata. See [PUBLICATION_STEPS.md](docs/PUBLICATION_STEPS.md).
