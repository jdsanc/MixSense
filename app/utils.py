# app/utils.py
"""
Shared utility functions for NMR Chemistry Analysis.
"""

import os
import re
import tempfile
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_spectrum(ppm: List[float], intensity: List[float], title: str, style: str = "auto") -> str:
    """
    Plot an NMR spectrum and save to a temporary PNG file.
    
    Args:
        ppm: List of chemical shift values (ppm)
        intensity: List of intensity values
        title: Title for the plot
        style: "sticks" for stick plot, "line" for continuous, "auto" to choose based on data
    
    Returns:
        Path to the saved PNG file
    """
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=150)

    if style == "sticks" or (style == "auto" and len(ppm) <= 100):
        for x, y in zip(ppm, intensity or [1.0] * len(ppm)):
            ax.vlines(x, 0, y, linewidth=1.2)
    else:
        ax.plot(ppm, intensity, linewidth=0.8)

    if len(ppm) >= 2:
        ax.invert_xaxis()
    ax.set_xlabel("ppm")
    ax.set_ylabel("intensity")
    ax.set_title(title, fontsize=10)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # Create unique filename
    safe_title = re.sub(r'[^A-Za-z0-9_.-]+', '_', title)[:40]
    path = os.path.join(tempfile.mkdtemp(), f"{safe_title}.png")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

