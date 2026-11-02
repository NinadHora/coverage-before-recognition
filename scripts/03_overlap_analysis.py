#!/usr/bin/env python3
"""Analyze overlap of missed frames between detectors.

Reports:
- Pairwise overlap of missed frames.
- Frames missed by all detectors (universal misses).
- Frames missed exclusively by each detector (architecture-exclusive misses).
- Demographic profile of architecture-exclusive misses.

Usage:
    python 03_overlap_analysis.py \\
        --results-dir results/detections \\
        --metadata configs/ccv2_metadata.csv \\
        --target yolo11_ft_conf0.25_imgsz640
"""
import argparse
from pathlib import Path

import pandas as pd


def load_missed(csv_path):
    df = pd.read_csv(csv_path)
    df["subject_id"] = df["subject_id"].astype(str).str.zfill(4)
    return set(df[df["detected"] == False]["frame_path"].values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--target", required=True,
                   help="Stem of the target detector CSV, "
                        "e.g. yolo11_ft_conf0.25_imgsz640")
    p.add_argument("--others", nargs="+", required=True,
                   help="Stems of other detector CSVs")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    target_set = load_missed(results_dir / f"{args.target}.csv")
    others = {
        name: load_missed(results_dir / f"{name}.csv")
        for name in args.others
    }

    print(f"\nTarget detector: {args.target}")
    print(f"Missed by target: {len(target_set)}")
    for name, s in others.items():
        print(f"Missed by {name}: {len(s)}")

    all_sets = [target_set] + list(others.values())
    universal = set.intersection(*all_sets) if all_sets else set()
    print(f"\nUniversal misses "
          f"(missed by every detector tested): {len(universal)}")

    others_union = set().union(*others.values()) if others else set()
    exclusive = target_set - others_union
    print(f"Target-exclusive misses "
          f"(missed only by {args.target}): {len(exclusive)}")

    if exclusive:
        df_target = pd.read_csv(results_dir / f"{args.target}.csv")
        df_target["subject_id"] = (df_target["subject_id"]
                                   .astype(str).str.zfill(4))
        lost = df_target[df_target["frame_path"].isin(exclusive)]
        subjects = lost["subject_id"].unique()
        print(f"Subjects affected by target-exclusive misses: {len(subjects)}")

        meta = pd.read_csv(args.metadata)
        meta["subject_id"] = meta["subject_id"].astype(str).str.zfill(4)
        meta_subj = (meta.drop_duplicates("subject_id")
                     .set_index("subject_id"))
        merged = lost.merge(
            meta_subj, left_on="subject_id",
            right_index=True, how="left")

        print("\nTarget-exclusive misses by Monk scale:")
        g = merged.groupby("monk_num").size().reset_index(name="exclusive")
        totals = (df_target.merge(
            meta_subj, left_on="subject_id",
            right_index=True, how="left")
                  .groupby("monk_num").size()
                  .reset_index(name="total"))
        g = g.merge(totals, on="monk_num")
        g["loss_rate_pct"] = (g["exclusive"] / g["total"] * 100).round(2)
        print(g.to_string(index=False))

        print("\nTarget-exclusive misses by gender:")
        print(merged.groupby("gender").size()
              .reset_index(name="exclusive").to_string(index=False))

        print("\nTarget-exclusive misses by country:")
        print(merged.groupby("country").size()
              .reset_index(name="exclusive").to_string(index=False))


if __name__ == "__main__":
    main()
