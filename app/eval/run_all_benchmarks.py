#!/usr/bin/env python3
"""
Master script to run all MixSense benchmarks.

This script runs both:
1. Legacy benchmarks (circular evaluation - for algorithm correctness)
2. NEW realistic evaluation (domain gap simulation - for publication)

Usage:
    python -m app.eval.run_all_benchmarks
    python -m app.eval.run_all_benchmarks --quick
    python -m app.eval.run_all_benchmarks --realistic-only  # Skip legacy, run publication eval
    python -m app.eval.run_all_benchmarks --output-dir results/
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_all(quick: bool = False, output_dir: str = "results", realistic_only: bool = False):
    """
    Run all benchmark suites.

    Args:
        quick: Run quick (smaller) benchmarks
        output_dir: Directory for output files
        realistic_only: If True, skip legacy benchmarks and only run publication-quality eval
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": "quick" if quick else "full",
        "realistic_only": realistic_only,
        "benchmarks": {},
    }

    # =========================================================================
    # PART A: LEGACY BENCHMARKS (Circular evaluation for algorithm correctness)
    # =========================================================================
    if not realistic_only:
        # ---------------------------------------------------------------------
        # 1. Generate Test Data
        # ---------------------------------------------------------------------
        print_header("1. [LEGACY] Generating Test Data")
        try:
            from app.eval.generate_test_data import generate_all_test_data
            test_data_dir = os.path.join(output_dir, "test_data")
            generate_all_test_data(test_data_dir)
            summary["benchmarks"]["test_data"] = {"status": "generated", "path": test_data_dir}
        except Exception as e:
            print(f"Error generating test data: {e}")
            summary["benchmarks"]["test_data"] = {"status": "error", "error": str(e)}

        # ---------------------------------------------------------------------
        # 2. NMRBank Tests
        # ---------------------------------------------------------------------
        print_header("2. [LEGACY] NMRBank Reference Tests")
        try:
            import pytest
            nmr_result_path = os.path.join(output_dir, f"nmrbank_tests_{timestamp}.xml")
            exit_code = pytest.main([
                "app/eval/test_nmrbank.py",
                "-v",
                "--tb=short",
                f"--junitxml={nmr_result_path}",
            ])
            summary["benchmarks"]["nmrbank_tests"] = {
                "status": "pass" if exit_code == 0 else "fail",
                "exit_code": exit_code,
                "output": nmr_result_path,
            }
        except Exception as e:
            print(f"Error running NMRBank tests: {e}")
            summary["benchmarks"]["nmrbank_tests"] = {"status": "error", "error": str(e)}

        # ---------------------------------------------------------------------
        # 3. Deconvolution Benchmark (Legacy - circular)
        # ---------------------------------------------------------------------
        print_header("3. [LEGACY] Deconvolution Benchmark (Circular)")
        try:
            from app.eval.benchmark_deconvolution import run_full_benchmark as run_deconv
            deconv_path = os.path.join(output_dir, f"deconvolution_{timestamp}.json")
            deconv_results = run_deconv(quick=quick, output_path=deconv_path)
            summary["benchmarks"]["deconvolution_legacy"] = {
                "status": "complete",
                "output": deconv_path,
                "summary": deconv_results.get("summary", {}),
            }
        except Exception as e:
            print(f"Error running deconvolution benchmark: {e}")
            summary["benchmarks"]["deconvolution_legacy"] = {"status": "error", "error": str(e)}

        # ---------------------------------------------------------------------
        # 4. Reaction Prediction Benchmark
        # ---------------------------------------------------------------------
        print_header("4. [LEGACY] Reaction Prediction Benchmark")
        try:
            from app.eval.benchmark_reactions import run_full_benchmark as run_rxn
            rxn_path = os.path.join(output_dir, f"reactions_{timestamp}.json")
            rxn_results = run_rxn(quick=quick, output_path=rxn_path)
            summary["benchmarks"]["reactions"] = {
                "status": "complete",
                "output": rxn_path,
                "summary": rxn_results.get("summary", {}),
            }
        except Exception as e:
            print(f"Error running reaction benchmark: {e}")
            summary["benchmarks"]["reactions"] = {"status": "error", "error": str(e)}

        # ---------------------------------------------------------------------
        # 5. End-to-End Pipeline Benchmark
        # ---------------------------------------------------------------------
        print_header("5. [LEGACY] End-to-End Pipeline Benchmark")
        try:
            from app.eval.benchmark_pipeline import run_full_pipeline_benchmark
            pipeline_path = os.path.join(output_dir, f"pipeline_{timestamp}.json")
            pipeline_results = run_full_pipeline_benchmark(quick=quick, output_path=pipeline_path)
            summary["benchmarks"]["pipeline"] = {
                "status": "complete",
                "output": pipeline_path,
                "success_rate": pipeline_results.get("success_rate", 0),
            }
        except Exception as e:
            print(f"Error running pipeline benchmark: {e}")
            summary["benchmarks"]["pipeline"] = {"status": "error", "error": str(e)}
    else:
        print_header("Skipping legacy benchmarks (--realistic-only mode)")

    # =========================================================================
    # PART B: REALISTIC EVALUATION (Domain gap - for publication)
    # =========================================================================

    # -------------------------------------------------------------------------
    # 6. Realistic Evaluation with Domain Gap
    # -------------------------------------------------------------------------
    print_header("6. [NEW] Realistic Evaluation (Domain Gap)")
    try:
        from app.eval.realistic_evaluation import (
            run_evaluation,
            DifficultyLevel,
        )

        realistic_results = {}
        difficulties = ["easy", "medium", "hard"] if not quick else ["medium"]

        for diff in difficulties:
            diff_level = DifficultyLevel(diff)
            print(f"\n  Running {diff.upper()} difficulty...")
            diff_results = run_evaluation(
                difficulty=diff_level,
                seed=42,
            )
            realistic_results[diff] = diff_results

        realistic_path = os.path.join(output_dir, f"realistic_eval_{timestamp}.json")
        with open(realistic_path, 'w') as f:
            json.dump(realistic_results, f, indent=2)

        # Extract summary
        realistic_summary = {}
        for diff, res in realistic_results.items():
            if "summary" in res:
                realistic_summary[diff] = {
                    "mae_mean": res["summary"].get("mae_mean", "N/A"),
                    "n_tests": res["summary"].get("n_tests", 0),
                }

        summary["benchmarks"]["realistic_evaluation"] = {
            "status": "complete",
            "output": realistic_path,
            "by_difficulty": realistic_summary,
        }
    except Exception as e:
        print(f"Error running realistic evaluation: {e}")
        import traceback
        traceback.print_exc()
        summary["benchmarks"]["realistic_evaluation"] = {"status": "error", "error": str(e)}

    # -------------------------------------------------------------------------
    # 7. Baseline Comparison (Publication benchmark)
    # -------------------------------------------------------------------------
    print_header("7. [NEW] Baseline Comparison (7 methods)")
    try:
        from app.eval.benchmark_with_baselines import run_benchmark, run_full_comparison
        from app.eval.spectrum_perturbation import DifficultyLevel

        if quick:
            # Quick mode: just medium difficulty
            baseline_results = run_benchmark(
                difficulty=DifficultyLevel.MEDIUM,
                seed=42,
            )
        else:
            # Full mode: all difficulties
            baseline_results = run_full_comparison(seed=42)

        baseline_path = os.path.join(output_dir, f"baseline_comparison_{timestamp}.json")
        with open(baseline_path, 'w') as f:
            json.dump(baseline_results, f, indent=2)

        # Extract ranking
        if "ranking" in baseline_results:
            ranking = baseline_results["ranking"][:3]  # Top 3
        elif "by_difficulty" in baseline_results:
            ranking = list(baseline_results["by_difficulty"].get("medium", {}).keys())[:3]
        else:
            ranking = []

        summary["benchmarks"]["baseline_comparison"] = {
            "status": "complete",
            "output": baseline_path,
            "top_methods": ranking,
        }
    except Exception as e:
        print(f"Error running baseline comparison: {e}")
        import traceback
        traceback.print_exc()
        summary["benchmarks"]["baseline_comparison"] = {"status": "error", "error": str(e)}

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print_header("FINAL SUMMARY")

    print(f"\nTimestamp: {summary['timestamp']}")
    print(f"Mode: {summary['mode']}")
    print(f"Output Directory: {output_dir}")

    print("\nBenchmark Results:")
    for name, result in summary["benchmarks"].items():
        status = result.get("status", "unknown")
        if status == "complete":
            if "summary" in result:
                detail = json.dumps(result["summary"], indent=None)[:60] + "..."
            elif "success_rate" in result:
                detail = f"success_rate={result['success_rate']:.1%}"
            else:
                detail = ""
            print(f"  [{status.upper():8}] {name}: {detail}")
        elif status == "error":
            print(f"  [ERROR   ] {name}: {result.get('error', 'unknown error')}")
        else:
            print(f"  [{status.upper():8}] {name}")

    # Save summary
    summary_path = os.path.join(output_dir, f"summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run all MixSense benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks")
    parser.add_argument("--output-dir", "-o", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--realistic-only", action="store_true",
                        help="Skip legacy benchmarks, only run publication-quality evaluation")

    args = parser.parse_args()

    run_all(quick=args.quick, output_dir=args.output_dir, realistic_only=args.realistic_only)


if __name__ == "__main__":
    main()
