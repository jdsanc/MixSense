"""Digitizer performance benchmark.

Runs the extraction library directly (no API server needed) against the
test case images. Outputs CSVs and overlay PNGs to test/digitizer/output/
for visual inspection.

Usage:
    # Env: mixsense
    conda run -n mixsense python test/digitizer/digitizer_benchmark.py
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / ".local/digitizer"))

from spectra.job_plan import parse_job_plan, execute_plan  # noqa: E402

CASES_DIR = Path(__file__).parent / "cases"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

CASES = [
    {
        "name": "01_pmma_raman_single_curve",
        "image": CASES_DIR / "01_pmma_raman.png",
        "job_plan": {
            "figure_title": "PMMA Raman",
            "panels": [{
                "panel_id": "main",
                "bounding_box": {"x_min": 70, "y_min": 55, "x_max": 715, "y_max": 458},
                "x_axis_label": "Raman shift (cm⁻¹)",
                "y_axis_label": "Intensity",
                "x_tick_min": 250, "x_tick_max": 3500,
                "y_tick_min": 0.0, "y_tick_max": 1.0,
                "spectrum_type": "Raman",
                # Calibration points at the first and last visible tick marks.
                # Pixel values below are bbox edges (conservative default).
                # Replace with exact tick pixel positions from the grid overlay.
                "x_calibration_points": [
                    {"pixel": 70, "value": 250},
                    {"pixel": 715, "value": 3500},
                ],
                "jobs": [{"job_id": "pmma", "label": "PMMA", "color_hint": "#1f77b4"}],
            }],
        },
    },
    {
        "name": "02_citric_acid_multi_curve",
        "image": CASES_DIR / "02_citric_acid.png",
        "job_plan": {
            "figure_title": "Citric Acid Raman",
            "panels": [{
                "panel_id": "main",
                "bounding_box": {"x_min": 197, "y_min": 49, "x_max": 954, "y_max": 610},
                "x_axis_label": "Raman shift (cm⁻¹)",
                "y_axis_label": "Raman intensity (a.u.)",
                "x_tick_min": 400, "x_tick_max": 1800,
                "y_tick_min": 0, "y_tick_max": 3000,
                "spectrum_type": "Raman",
                "x_calibration_points": [
                    {"pixel": 197, "value": 400},
                    {"pixel": 954, "value": 1800},
                ],
                "jobs": [
                    {
                        "job_id": "aqueous",
                        "label": "citric acid aqueous solution",
                        "color_hint": "#000000",
                        "roi": {"x_min": 197, "y_min": 49, "x_max": 954, "y_max": 440},
                    },
                    {
                        "job_id": "solid",
                        "label": "citric acid solid",
                        "color_hint": "#cd4440",
                        "roi": {"x_min": 197, "y_min": 300, "x_max": 954, "y_max": 610},
                    },
                ],
            }],
        },
    },
    {
        "name": "04_stacked_raman_spectra",
        "image": CASES_DIR / "03_stacked_raman.png",
        "job_plan": {
            "figure_title": "Stacked Raman Spectra",
            "panels": [{
                "panel_id": "main",
                "bounding_box": {"x_min": 46, "y_min": 50, "x_max": 1155, "y_max": 643},
                "x_axis_label": "Raman shift (cm⁻¹)",
                "y_axis_label": "Intensity (a.u.)",
                "x_tick_min": 200, "x_tick_max": 3500,
                "y_tick_min": 0, "y_tick_max": 1,
                "y_calibration": "per_curve_normalized",
                "spectrum_type": "Raman",
                "x_calibration_points": [
                    {"pixel": 46, "value": 200},
                    {"pixel": 1155, "value": 3500},
                ],
                "jobs": [
                    {
                        "job_id": "polyethylene",
                        "label": "Polyethylene",
                        "color_hint": "#1f77b4",
                        "roi": {"x_min": 46, "y_min": 349, "x_max": 1155, "y_max": 643},
                    },
                    {
                        "job_id": "polystyrene",
                        "label": "Polystyrene",
                        "color_hint": "#d62728",
                        "roi": {"x_min": 46, "y_min": 258, "x_max": 1155, "y_max": 438},
                    },
                    {
                        "job_id": "nylon",
                        "label": "Nylon 6,6",
                        "color_hint": "#2ca02c",
                        "roi": {"x_min": 46, "y_min": 50, "x_max": 1155, "y_max": 259},
                    },
                ],
            }],
        },
    },
]


def save_overlay(image_path: Path, results, case_name: str) -> None:
    """Draw extracted points over the source image and save as PNG."""
    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: source image with bounding box
    axes[0].imshow(img_rgb)
    axes[0].set_title("Source image")
    axes[0].axis("off")

    # Right: extracted curves
    axes[1].set_title("Extracted curves")
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    for i, jr in enumerate(results.job_results):
        res = jr.result
        if res.error:
            print(f"  [{jr.job_id}] ERROR: {res.error}")
            continue
        color = colors[i % len(colors)]
        axes[1].plot(res.x_data, res.y_data, color=color, lw=1.5, label=jr.label or jr.job_id)
        d = res.diagnostics
        print(
            f"  [{jr.job_id}] n={res.n_points}  coverage={res.x_coverage:.2f}"
            f"  quality={res.quality_score:.3f}"
            f"  adherence={d.pixel_adherence:.3f}  clusters={d.column_cluster_score:.3f}"
        )

    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("X (data units)")
    axes[1].set_ylabel("Y (data units)")
    plt.tight_layout()

    out_path = OUT_DIR / f"{case_name}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  overlay → {out_path.relative_to(ROOT)}")


def save_csv(results, case_name: str) -> None:
    for jr in results.job_results:
        res = jr.result
        if res.error or len(res.x_data) == 0:
            continue
        out_path = OUT_DIR / f"{case_name}_{jr.job_id}.csv"
        lines = ["x,y"] + [f"{x},{y}" for x, y in zip(res.x_data.tolist(), res.y_data.tolist())]
        out_path.write_text("\n".join(lines))
        print(f"  csv    → {out_path.relative_to(ROOT)}")


def main() -> None:
    for case in CASES:
        name = case["name"]
        image_path = case["image"]
        print(f"\n{'='*60}")
        print(f"Case: {name}")

        if not image_path.exists():
            print(f"  SKIP — image not found: {image_path}")
            continue

        img_bgr = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        plan = parse_job_plan(case["job_plan"])
        results = execute_plan(plan, img_rgb)

        print(
            f"  jobs={results.total_jobs}  ok={results.successful_jobs}"
            f"  mean_quality={results.mean_quality:.3f}"
            f"  elapsed={results.elapsed_s:.2f}s"
        )

        save_overlay(image_path, results, name)
        save_csv(results, name)

    print(f"\nAll outputs in: test/digitizer/output/")


if __name__ == "__main__":
    main()
