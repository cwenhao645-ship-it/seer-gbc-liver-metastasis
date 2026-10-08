# Public-snapshot relationship to the peer-review package

This public folder was derived from the authors' existing round-2 peer-review
reproducibility ZIP and the final Figure 1, 3, 4, and 5 TIFF files. The source ZIP
and submission documents were not modified. Its SHA-256 is recorded in
`provenance/RELEASE_PROVENANCE.json` without publishing the author's local paths.

## Unchanged research artifacts

All seven supplied analysis scripts and the model-designation correction diff
are byte-identical to their source ZIP entries. Candidate tuning settings,
data-dictionary definitions, aggregate model metrics, and the recoding/redevelopment
results are unchanged. No models were fitted and no analyses were rerun during
public packaging.

The original tuning workbook contains 12 records: three tiers by four models.
The additional CSV and JSON are structured exports of those same records.

## Public disclosure subset

- In `aggregate_outputs/target_outcome_status_SMD.csv`, the categorical age-group
  and race/ethnicity count blocks are omitted in their entirety. This avoids
  exposing small demographic counts or recovering them by subtracting the
  remaining categories from the cohort total. Continuous age summaries and the
  other retained rows are unchanged.
- The matching categorical age-group and race balance rows are omitted from
  `aggregate_outputs/covariate_balance.csv`, because exact balance statistics can
  also reveal suppressed category counts. The public data dictionary retains
  all 21 variable definitions, but omits the age-group frequency description.
- In `aggregate_outputs/demographic_subgroup_performance.csv`, all confusion-cell
  columns and sensitivity/specificity/PPV/NPV estimates and their confidence
  intervals are omitted across every subgroup. Retaining complementary
  subgroups or precise threshold metrics could reveal small demographic
  confusion counts. N, event counts, prevalence, ROC-AUC, PR-AUC, Brier score,
  calibration summaries, and their available uncertainty summaries remain.
- The matching subgroup bootstrap file retains only domain, subgroup,
  replicate, ROC-AUC, PR-AUC, and Brier score.
- Source SVG/PDF case-point SHAP graphics and their TIFF equivalents are not
  copied to this public snapshot. Aggregated categorical and binned-median
  SHAP source summaries are retained. Figures 1, 3, 4, and 5 are copied unchanged.
- Duplicated figure-source CSVs in the peer-review ZIP are represented once,
  under `aggregate_outputs/figure_source/`. Curve source files already omit
  individual prediction thresholds; the DCA threshold is a prespecified grid,
  not a list of case-level scores.
- The peer-review README is replaced by a public README with verified CLI
  paths, licensing scope, and limits on reproduction. Historical method
  clarification remains, and stale paths in the public code-trace note are
  redirected to the supplied files.

The file-level mapping and disposition are listed in
`provenance/SOURCE_TO_PUBLIC.csv`. The hash manifest covers the public files,
not the original confidential/private working directory. The manifest excludes
itself to avoid a circular checksum.

This is a conservative public-release disclosure screen, not a blanket legal
certification. It does not make the public subset identical to the peer-review
materials or replace the authors' applicable SEER agreement.
