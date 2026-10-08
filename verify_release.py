"""Check the public snapshot without accessing patient-level SEER data."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
MANIFEST = "provenance/SHA256_MANIFEST.csv"
BLOCKED_EXTENSIONS = {".pkl", ".pickle", ".joblib", ".cbm", ".npy", ".npz", ".parquet", ".feather", ".sav", ".dta", ".zip", ".docx"}
BLOCKED_HEADERS = {"patient_id", "patient id", "record_id", "case_id", "y_true", "pred_prob", "locked_probability"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distributed_files() -> set[str]:
    return {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in {".git", ".venv", "venv", "__pycache__"} for part in path.relative_to(ROOT).parts)
        and path.suffix not in {".pyc", ".pyo"}
        and path.relative_to(ROOT).as_posix() != MANIFEST
    }


def main() -> int:
    errors = []
    manifest = ROOT / MANIFEST
    if not manifest.is_file():
        print("FAIL: missing checksum manifest")
        return 1
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {row["path"] for row in rows}
    if len(expected) != len(rows):
        errors.append("Duplicate manifest paths")
    actual = distributed_files()
    for name in sorted(expected - actual):
        errors.append("Missing file: " + name)
    for name in sorted(actual - expected):
        errors.append("Unlisted file: " + name)
    for row in rows:
        path = (ROOT / row["path"]).resolve()
        if not path.is_relative_to(ROOT):
            errors.append("Manifest path outside repository")
            continue
        if not path.is_file():
            continue
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            errors.append("Changed file: " + row["path"])
        if path.suffix.lower() in BLOCKED_EXTENSIONS:
            errors.append("Restricted/unexpected artifact: " + row["path"])
        try:
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=row["path"])
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8-sig"))
            elif path.suffix == ".csv" and "/aggregate_outputs/" in path.as_posix():
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    headers = next(csv.reader(stream), [])
                if BLOCKED_HEADERS.intersection(header.lower().strip() for header in headers):
                    errors.append("Case-level header: " + row["path"])
        except (ValueError, SyntaxError, UnicodeError) as exc:
            errors.append(f"Invalid file {row['path']}: {exc}")
    parameters = ROOT / "model_metadata/candidate_model_parameters.json"
    if parameters.is_file():
        records = json.loads(parameters.read_text(encoding="utf-8"))
        pairs = {(record["Tier"], record["Model"]) for record in records}
        if len(records) != 12 or len(pairs) != 12:
            errors.append("Expected 12 distinct tier/model parameter records")
    dictionary = ROOT / "model_metadata/data_dictionary_TableS8.csv"
    if dictionary.is_file():
        with dictionary.open(encoding="utf-8-sig", newline="") as stream:
            if len(list(csv.DictReader(stream))) != 21:
                errors.append("Expected 21 data-dictionary records")
    if errors:
        print("FAIL")
        print("\n".join(errors))
        return 1
    print(f"PASS: {len(rows)} files match their checksums; syntax and public-snapshot structure checked.")
    print("No statistical rerun or legal-compliance certification was performed by this check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
