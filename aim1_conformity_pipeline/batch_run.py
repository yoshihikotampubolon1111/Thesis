"""Batch runner for Aim 1 cage-to-endplate conformity analyses."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from aim1_conformity import DEFAULT_TOLERANCES_MM, run_conformity_analysis


REQUIRED_COLUMNS = {
    "case_id",
    "cage_stl",
    "endplate_stl",
    "transform_file",
    "group",
    "level",
    "endplate_side",
    "cage_model",
}


def _resolve_path(path_text: Any, base_dir: Path, required: bool = True) -> Path | None:
    """Resolve a CSV path entry, treating empty optional values as ``None``."""

    if pd.isna(path_text) or str(path_text).strip() == "":
        if required:
            raise ValueError("Required path entry is empty.")
        return None

    path = Path(str(path_text).strip())
    if path.is_absolute():
        return path
    return base_dir / path


def run_batch(
    cases_csv: str | Path,
    output_dir: str | Path,
    samples: int = 100_000,
    tolerances: list[float] | None = None,
    rim_width_mm: float | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all cases listed in a batch CSV and save a combined summary CSV."""

    cases_path = Path(cases_csv)
    if not cases_path.exists():
        raise FileNotFoundError(f"Cases CSV does not exist: {cases_path}")
    if not cases_path.is_file():
        raise ValueError(f"Cases path is not a file: {cases_path}")

    cases = pd.read_csv(cases_path)
    missing = REQUIRED_COLUMNS.difference(cases.columns)
    if missing:
        raise ValueError(
            f"Cases CSV is missing required columns: {', '.join(sorted(missing))}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    base_dir = cases_path.parent
    tolerances = tolerances if tolerances is not None else list(DEFAULT_TOLERANCES_MM)

    metrics_rows: list[dict[str, Any]] = []
    for row_number, row in cases.iterrows():
        case_id = str(row["case_id"]).strip()
        if not case_id:
            raise ValueError(f"case_id is empty at CSV row {row_number + 2}")

        cage_path = _resolve_path(row["cage_stl"], base_dir, required=True)
        endplate_path = _resolve_path(row["endplate_stl"], base_dir, required=True)
        transform_path = _resolve_path(row["transform_file"], base_dir, required=False)
        assert cage_path is not None
        assert endplate_path is not None

        metadata = {
            "group": row["group"],
            "level": row["level"],
            "endplate_side": row["endplate_side"],
            "cage_model": row["cage_model"],
        }

        case_output_dir = output_path / case_id
        print(f"Running case {case_id} -> {case_output_dir}")
        metrics = run_conformity_analysis(
            case_id=case_id,
            cage_path=cage_path,
            endplate_path=endplate_path,
            output_dir=case_output_dir,
            transform_path=transform_path,
            samples=samples,
            tolerances=tolerances,
            rim_width_mm=rim_width_mm,
            seed=seed,
            metadata=metadata,
        )
        metrics_rows.append(metrics)

    summary = pd.DataFrame(metrics_rows)
    summary_path = output_path / "all_cases_summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Batch summary written to: {summary_path}")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """Create command-line parser for batch analysis."""

    parser = argparse.ArgumentParser(
        description="Run Aim 1 geometric conformity analysis for a CSV batch."
    )
    parser.add_argument(
        "--cases",
        required=True,
        help="Batch CSV with case_id,cage_stl,endplate_stl,transform_file,group,level,endplate_side,cage_model.",
    )
    parser.add_argument("--output", required=True, help="Batch output directory.")
    parser.add_argument(
        "--samples",
        type=int,
        default=100_000,
        help="Number of area-weighted cage surface samples per case.",
    )
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=list(DEFAULT_TOLERANCES_MM),
        help="Distance tolerances in mm for proximity-based contact area fractions.",
    )
    parser.add_argument(
        "--rim_width_mm",
        type=float,
        default=None,
        help="Optional approximate rim width in mm for PCA/convex-hull rim overlap.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible surface sampling.",
    )
    return parser


def main() -> None:
    """Command-line entrypoint."""

    parser = _build_parser()
    args = parser.parse_args()
    run_batch(
        cases_csv=args.cases,
        output_dir=args.output,
        samples=args.samples,
        tolerances=args.tolerances,
        rim_width_mm=args.rim_width_mm,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
