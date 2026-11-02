#!/usr/bin/env python3
"""Compute stratified coverage from a detections CSV.

Produces global, univariate (gender, Monk, age), bivariate
(gender x Monk), and trivariate (gender x Monk x age) coverage tables.

Usage:
    python 02_analyze_coverage.py \\
        --detections results/detections/yolo11_ft_conf0.25_imgsz640.csv \\
        --metadata configs/ccv2_metadata.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def load_merged(detections_csv, metadata_csv):
    df = pd.read_csv(detections_csv)
    df["subject_id"] = df["subject_id"].astype(str).str.zfill(4)

    meta = pd.read_csv(metadata_csv)
    meta["subject_id"] = meta["subject_id"].astype(str).str.zfill(4)
    meta_subj = meta.drop_duplicates("subject_id").set_index("subject_id")

    return df.merge(meta_subj, left_on="subject_id",
                    right_index=True, how="left")


def coverage_by(merged, cols):
    g = merged.groupby(cols).agg(
        n=("detected", "count"),
        det=("detected", "sum"),
    ).reset_index()
    g["coverage_pct"] = (g["det"] / g["n"] * 100).round(2)
    g["missed"] = g["n"] - g["det"]
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detections", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--min-cell", type=int, default=10,
                   help="Minimum cell size for trivariate output")
    args = p.parse_args()

    merged = load_merged(args.detections, args.metadata)
    total = len(merged)
    detected = int(merged["detected"].sum())
    print(f"\nDetections file: {Path(args.detections).name}")
    print(f"Global coverage: {detected / total * 100:.2f}% "
          f"({detected} of {total} frames)")

    print("\nCoverage by gender:")
    print(coverage_by(merged, ["gender"]).to_string(index=False))

    print("\nCoverage by Monk scale:")
    print(coverage_by(merged, ["monk_num"]).to_string(index=False))

    print("\nCoverage by age group:")
    print(coverage_by(merged, ["age_group"]).to_string(index=False))

    print("\nCoverage by gender x Monk, cells with coverage below 95%:")
    bi = coverage_by(merged, ["gender", "monk_num"])
    low = bi[bi["coverage_pct"] < 95].sort_values("coverage_pct")
    print(low.to_string(index=False))

    print(f"\nCoverage by gender x Monk x age group "
          f"(cells with n >= {args.min_cell}, coverage below 80%):")
    tri = coverage_by(merged, ["gender", "monk_num", "age_group"])
    tri_low = tri[(tri["n"] >= args.min_cell)
                  & (tri["coverage_pct"] < 80)].sort_values("coverage_pct")
    print(tri_low.to_string(index=False))


if __name__ == "__main__":
    main()
