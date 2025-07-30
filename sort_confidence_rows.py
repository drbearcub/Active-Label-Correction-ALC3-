#!/usr/bin/env python
"""Sort rows in probability.jsonl by confidence metrics and write sorted JSONL.

By default creates two files alongside the source file:
    <src>_prob_product_sorted.jsonl
    <src>_geo_mean_sorted.jsonl

You can choose a different input or metrics via CLI.

Usage:
    python sort_confidence_rows.py --file probability.jsonl \
           --metrics prob_product geo_mean \
           --descending   # sort from highest to lowest (default)

    # Specify output directory
    python sort_confidence_rows.py --file probability.jsonl --out_dir sorted
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

DEFAULT_FILE = Path("results/probability.jsonl")
DEFAULT_METRICS = ["prob_product", "geo_mean"]


def load_rows(jsonl_path: Path) -> List[Dict]:
    rows = []
    with jsonl_path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed JSON at line {line_num}: {e}")
    return rows


def write_sorted(rows: List[Dict], metric: str, out_path: Path, descending: bool = True):
    # Filter out rows missing the metric or non-numeric values
    filtered = [r for r in rows if isinstance(r.get(metric), (int, float))]
    sorted_rows = sorted(filtered, key=lambda r: r[metric], reverse=descending)

    with out_path.open("w") as f:
        for obj in sorted_rows:
            json.dump(obj, f)
            f.write("\n")
    print(f"Wrote {len(sorted_rows)} rows sorted by '{metric}' to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Sort probability.jsonl rows by confidence metrics.")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="Path to probability.jsonl input file")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="Metrics to sort by")
    parser.add_argument("--out_dir", type=Path, help="Directory to store sorted files (default: alongside input)")
    parser.add_argument("--ascending", action="store_true", help="Sort in ascending order instead of descending")
    args = parser.parse_args()

    if not args.file.exists():
        raise FileNotFoundError(f"{args.file} not found")

    rows = load_rows(args.file)
    descending = not args.ascending

    out_base_dir = args.out_dir if args.out_dir else args.file.parent
    out_base_dir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        out_path = out_base_dir / f"{args.file.stem}_{metric}_sorted.jsonl"
        write_sorted(rows, metric, out_path, descending=descending)


if __name__ == "__main__":
    main() 