# SEER access and excluded materials

The manuscript cites SEER Research Plus Data, 17 Registries, November 2024
submission, 2000-2022. The study window is 2010-2022. The source extract and
case-level derivatives are not distributed here.

SEER explains that the restriction on releasing raw or individual-level data
does not generally prohibit sharing calculated summary statistics:

- [Data agreements and summary-statistic clarification](https://seer.cancer.gov/data/agreements.html)
- [November 2024 SEER Research Data Use Agreement](https://seer.cancer.gov/data-software/documentation/seerstat/nov2024/seer-dua-nov2024.html)
- [Current data-access policy](https://seer.cancer.gov/data/access-policy.html)
- [Access application and SEER*Stat instructions](https://seer.cancer.gov/data/access.html)

Eligibility differs between Research and Research Plus products. Current policy
restricts Research Plus access for specified locations. This repository cannot
grant access or promise that every reader can obtain the same extract. Readers
must use an authorized extract and the agreement applicable to their access.

The November 2024 agreement requires suppression of geography/demographic
statistics based on counts 1-4. The public snapshot therefore omits selected
demographic count blocks and subgroup threshold metrics, including values that
would allow suppressed counts to be reconstructed. See
[PUBLIC_RELEASE_NOTES.md](PUBLIC_RELEASE_NOTES.md).

Excluded materials include:

- raw SEER exports and case listings;
- model-ready case-level tables and patient identifiers;
- case-level outcomes paired with predictions;
- individual SHAP records and machine-readable case-point graphics;
- serialized trained estimators;
- private review correspondence, submission documents, credentials, and local logs.

The Python code necessarily mentions source-column names such as `Patient_ID`.
A column name in source code or a data dictionary is not a released patient
record. Running the code may generate restricted files locally. Never upload
the resulting output directories wholesale.
