"""Launch an authorized local rerun with the archived development settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PIPELINES = {
    "archived": "analysis_code/archived_pipeline/_embedded_modelopt_calibrated_ARCHIVED.py",
    "corrected": "analysis_code/corrected_pipeline/_embedded_modelopt_calibrated_DEVELOPMENT_OOF_CORRECTED.py",
}


def outside_repository(path: Path) -> bool:
    return not path.is_relative_to(ROOT) and not ROOT.is_relative_to(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Authorized SEER CSV outside the repository")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dedicated private directory outside the repository")
    parser.add_argument("--pipeline", choices=PIPELINES, default="archived")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without reading data or running analysis")
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not outside_repository(source) or not outside_repository(output):
        parser.error("Input and output must be outside the repository; output must not be its ancestor.")
    if source.suffix.lower() != ".csv":
        parser.error("Supply an explicit SEER CSV file.")
    if source.is_relative_to(output):
        parser.error("Keep the input extract outside the generated-output directory.")
    metadata = json.loads((ROOT / "model_metadata/archived_run_manifest_sanitized.json").read_text(encoding="utf-8-sig"))
    settings = metadata["arguments"]
    command = [sys.executable, str(ROOT / PIPELINES[args.pipeline]), "--workdir", str(source.parent), "--input", str(source), "--outdir", str(output)]
    for name, value in settings.items():
        if name in {"workdir", "input", "outdir", "threshold_rule_display"}:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    command.extend(["--threshold-rule", "sensitivity80", "--selection-metric", "PR-AUC"])
    print("Mode:", args.pipeline)
    print("Outputs may contain restricted records, predictions, SHAP values, and models. Do not upload them.")
    print("Command:", subprocess.list2cmdline(command))
    if args.dry_run:
        return 0
    if not source.is_file():
        parser.error("The authorized input CSV does not exist.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Choose a new or empty output directory; existing analysis artifacts will not be overwritten.")
    return subprocess.run(command, cwd=source.parent, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
