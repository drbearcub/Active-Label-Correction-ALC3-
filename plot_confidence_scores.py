#!/usr/bin/env python
"""Plot confidence score distributions from probability.jsonl.

Generates a 2×2 grid of histograms for the following metrics:
• Product of probabilities (prob_product)
• Geometric mean (geo_mean)
• Perplexity (perplexity)
• Average log probability (avg_log_prob)

Usage:
    python plot_confidence_scores.py [--file probability.jsonl] [--out scores.png]

If --out is not provided, the plots will be shown interactively.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_JSONL = Path("probability.jsonl")


def load_metrics(jsonl_path: Path) -> Dict[str, List[float]]:
    """Return dict mapping metric → list of values from all rows."""
    metrics = {
        "prob_product": [],
        "geo_mean": [],
        "perplexity": [],
        "avg_log_prob": [],
    }

    def _nan_const(_):
        """Return NaN for non-standard JSON constants (NaN, Infinity)."""
        return float("nan")

    with jsonl_path.open() as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line, parse_constant=_nan_const)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed JSON at line {line_num}: {e}")
                continue
            for key in metrics:
                value = obj.get(key)
                if isinstance(value, (int, float)):
                    metrics[key].append(value)
    return metrics


def make_plots(metrics: Dict[str, List[float]], out_dir: Optional[Path] = None):
    """Generate separate histogram plots for each metric.

    If `out_dir` is provided, each plot is saved as `<metric>.png` inside that
    directory. Otherwise the plots are shown interactively one after another.
    """

    sns.set(style="whitegrid")

    titles = {
        "prob_product": "Product of Probabilities",
        "geo_mean": "Geometric Mean of Probabilities",
        "perplexity": "Perplexity",
        "avg_log_prob": "Average Log Probability",
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for key, data in metrics.items():
        if not data:
            continue

        plt.figure(figsize=(6, 4))
        sns.histplot(data, kde=True, bins=30, color="steelblue")
        plt.title(titles.get(key, key))
        plt.xlabel(key)
        plt.ylabel("Count")
        plt.tight_layout()

        if out_dir is not None:
            file_path = out_dir / f"{key}.png"
            plt.savefig(file_path, dpi=300)
            print(f"Saved {file_path}")
            plt.close()
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot confidence score distributions from probability.jsonl")
    parser.add_argument("--file", type=Path, default=DEFAULT_JSONL, help="Path to probability.jsonl")
    parser.add_argument("--out_dir", type=Path, help="Directory to save individual metric plots. If omitted, plots are shown interactively.")
    args = parser.parse_args()

    if not args.file.exists():
        raise FileNotFoundError(f"{args.file} not found")

    metrics = load_metrics(args.file)
    make_plots(metrics, out_dir=args.out_dir)


if __name__ == "__main__":
    main() 