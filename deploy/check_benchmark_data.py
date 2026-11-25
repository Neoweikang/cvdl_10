"""Validate a labelled benchmark CSV without importing the OCR model stack.

Usage: python check_benchmark_data.py path/to/lines.csv
"""
import argparse
from collections import Counter
from pathlib import Path

from benchmark_data import load_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', type=Path, help='CSV with image_path,text,language columns')
    args = parser.parse_args(argv)

    rows = load_rows(args.csv_path)
    counts = Counter(row['language'] for row in rows)
    print(f'Benchmark data is valid: {len(rows)} labelled lines.')
    print(f"English (en): {counts['en']}")
    print(f"Chinese (zh): {counts['zh']}")


if __name__ == '__main__':
    main()
