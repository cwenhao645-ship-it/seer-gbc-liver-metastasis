# GitHub upload and publication archive

## Upload the prepared snapshot

1. Create a GitHub repository, for example `seer-gbc-liver-metastasis`. Leave the
   automatic README, license, and `.gitignore` options unchecked because these
   files are already supplied. Select Public only when ready to make it public.
2. Use **uploading an existing file**, or **Add file > Upload files**. Upload the
   contents of the prepared folder, including its subfolders and dotfiles. Keep
   `README.md`, `LICENSE`, `CITATION.cff`, and `requirements.txt` at repository
   root, not inside an extra `Discover_Oncology_GitHub_READY/` wrapper.
3. Verify that the root files and all nested folders appear. GitHub browser
   uploads do not apply `.gitignore`; upload only this prepared folder, not your
   project directory, private inputs, generated outputs, or the peer-review ZIP.
4. Commit with a factual message such as `Add minor-revision reproducibility
   snapshot`. Download the resulting repository and run `python verify_release.py`
   to check the upload. A folder is not evidence of a successful online upload.

GitHub currently permits up to 100 files in one browser upload and limits each
file to 25 MiB. The prepared snapshot is checked against both limits.

[Official GitHub file-upload instructions](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)

## Assign an archive DOI

1. Connect the author's GitHub account to Zenodo and enable this repository.
2. Check author names, rights/licensing for this mixed-material archive, and the
   description before creating a release. MIT applies to code, not SEER data or
   all accompanying figures/data. Do not assign invented ORCIDs or an article DOI.
3. Create a release such as `v1.0.0-rc1` for this pre-publication snapshot. Once
   Zenodo creates the record, verify that the record is public, its files match
   this snapshot, and its DOI resolves. The local folder itself has no DOI.
4. If you edit citation metadata or any file, regenerate the checksum manifest
   before creating a later version. Cite the version-specific archive DOI for
   the snapshot underlying the published article.

[Zenodo: enable a repository](https://help.zenodo.org/docs/github/enable-repository/)

[Zenodo: archive a release](https://help.zenodo.org/docs/github/archive-software/github-upload/)

## Synchronize the article once real identifiers exist

Add the actual GitHub URL and version-specific DOI to the manuscript's Code
Availability statement and, where appropriate, Data Availability statement.
Synchronize the clean and marked versions and the submission-system declaration.
Keep the individual-level SEER restriction and the exploratory/nonclinical
interpretation. A possible wording, after replacing the fields with real links,
is:

> The analysis code and accompanying aggregate research materials are available
> at [actual GitHub URL] and archived as version [actual version] at
> [actual Zenodo DOI]. Individual-level SEER data, case-level predictions and
> SHAP values, and fitted estimators are not redistributed. SEER source-data
> access is governed by the applicable SEER data-use requirements.

That example is an instruction, not a claim that public records already exist.
No manuscript or online declaration was changed during local folder preparation.
