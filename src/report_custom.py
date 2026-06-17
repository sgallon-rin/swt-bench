"""
Custom report script for arbitrary instance subsets.

Unlike src/report.py which uses fixed dataset sizes (300 for lite, etc.),
this script accepts a custom --total parameter to compute percentages
based on your actual subset size.

Usage:
    # Basic usage with custom total
    python -m src.report_custom run_instance_swt_logs/gold-subset-a/gold --total 52

    # With custom name
    python -m src.report_custom run_instance_swt_logs/gold-subset-a/gold --total 52 --name "My Subset"

    # With LaTeX output
    python -m src.report_custom run_instance_swt_logs/gold-subset-a/gold --total 52 --format latex

    # With coverage delta comparison (requires a gold run in the same parent directory)
    python -m src.report_custom run_instance_swt_logs/gold-subset-a/gold --total 52 --gold-run-id gold-subset-a
"""

import sys
from pathlib import Path
from typing import Tuple, Literal

from tabulate import tabulate
import fire

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figures.util import (
    collect_reports,
    no_error_count,
    ftp_count,
    actual_ftp_count,
    ftx_count,
    ptp_count,
    count_coverage_delta_gold,
    sum_coverage_delta,
    filtered_by_resolved,
)


def main(
    path: str,
    total: int,
    name: str = None,
    format: str = "github",
    gold_run_id: str = None,
):
    """
    Generate a report table for a custom instance subset.

    Args:
        path: Path to the model run directory under run_instance_swt_logs/{run_id}/{model}/
              e.g. "run_instance_swt_logs/gold-subset-a/gold"
        total: Total number of instances in your subset (used as denominator for percentages)
        name: Display name for the method (default: extracted from path)
        format: Output format - "github" for markdown table, "latex" for LaTeX
        gold_run_id: Optional run_id for gold predictions to compute coverage delta.
                     If not provided, coverage delta section is skipped.
    """
    instance_log_path = Path(path)
    if not instance_log_path.exists():
        raise FileNotFoundError(f"Instance log directory not found at {instance_log_path}")

    if name is None:
        name = instance_log_path.parent.name

    run_id = instance_log_path.parent.name
    model = instance_log_path.name

    reports = collect_reports(model, run_id, instance_log_path.parent.parent)

    fields = (
        [r"{$\mathcal{W}$}", r"{$\suc$}", r"{\ftx}", r"{\ftp}", r"{\ptp}"]
        if format.startswith("latex") else
        ["Applicability (W)", "Success Rate (S)", "F->X", "F->P", "P->P"]
    )

    applied = 100 * no_error_count(reports) / total
    success = 100 * ftp_count(reports) / total
    ftx = 100 * ftx_count(reports) / total
    ftp = 100 * actual_ftp_count(reports) / total
    ptp = 100 * ptp_count(reports) / total

    rows = [["Method", name]]
    rows.extend([key, f"{val:.1f}"] for key, val in zip(fields, [applied, success, ftx, ftp, ptp]))

    # Coverage delta section
    if gold_run_id:
        try:
            gold_reports = collect_reports("gold", gold_run_id, instance_log_path.parent.parent)
        except Exception:
            print(f"Warning: Gold run '{gold_run_id}' not found, skipping coverage delta")
            gold_reports = None

        if gold_reports:
            fields_delta = (
                [r"{$\dc^{\text{all}}$}", r"{$\dc^{\suc}$}", r"{$\dc^{\neg\suc}$}"]
                if format.startswith("latex") else
                ["Coverage Delta (Δᵃˡˡ)", "Coverage Delta Resolved (Δᔆ)", "Coverage Delta Unresolved (Δⁿᵒᵗ ᔆ)"]
            )
            total_coverage_possible = count_coverage_delta_gold(gold_reports)
            resolved_reports, unresolved_reports = filtered_by_resolved(reports)
            total_coverage_delta = 100 * sum_coverage_delta(reports) / total_coverage_possible if total_coverage_possible > 0 else 0
            countable_resolved = count_coverage_delta_gold(resolved_reports)
            resolved_coverage_delta = 100 * sum_coverage_delta(resolved_reports) / countable_resolved if countable_resolved > 0 else 0
            unresolved_coverage_delta = 100 * sum_coverage_delta(unresolved_reports) / (total_coverage_possible - countable_resolved) if total_coverage_possible - countable_resolved > 0 else 0
            rows.extend([key, f"{val:.1f}"] for key, val in zip(fields_delta, [total_coverage_delta, resolved_coverage_delta, unresolved_coverage_delta]))

    print(tabulate(rows, tablefmt=format, floatfmt=".1f"))


if __name__ == "__main__":
    fire.Fire(main)
