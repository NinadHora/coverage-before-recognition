#!/usr/bin/env python3
"""Prepare CCv2 metadata and frame manifest.

Reads the CCv2 annotations JSON and produces:
- configs/ccv2_metadata.csv: one row per video with demographic attributes.
- configs/frame_manifest.csv: one row per extracted frame.

Usage:
    python 00_prep_metadata.py \\
        --annotations /path/to/CasualConversationsV2.json \\
        --frames /path/to/frames \\
        --out-dir /path/to/configs
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def parse_monk(scale_str):
    if scale_str is None:
        return None
    try:
        return int(str(scale_str).replace("scale", "").strip())
    except Exception:
        return None


def parse_fitzpatrick(type_str):
    if type_str is None:
        return None
    mapping = {
        "type i": "I", "type ii": "II", "type iii": "III",
        "type iv": "IV", "type v": "V", "type vi": "VI",
    }
    return mapping.get(str(type_str).strip().lower())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True,
                   help="Path to CasualConversationsV2.json")
    p.add_argument("--frames", required=True,
                   help="Path to the folder containing per-subject frame subfolders")
    p.add_argument("--out-dir", required=True, help="Output folder for CSVs")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.annotations) as f:
        data = json.load(f)

    rows = []
    for entry in data:
        rows.append({
            "video_name": entry["video_name"],
            "subject_id": str(entry["subject_id"]).zfill(4),
            "age": entry.get("age"),
            "gender": entry.get("gender"),
            "native_language": entry.get("native_language"),
            "country": entry.get("geo_location", {}).get("country"),
            "state_region": entry.get("geo_location", {}).get("state_region"),
            "fitzpatrick": parse_fitzpatrick(
                entry.get("fitzpatrick_skin_tone", {}).get("type")),
            "monk_num": parse_monk(
                entry.get("monk_skin_tone", {}).get("scale")),
            "video_quality": entry.get("video_quality"),
            "capture_env": entry.get("capture_environment"),
        })

    df = pd.DataFrame(rows)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 55, 65, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
    )
    df.to_csv(out_dir / "ccv2_metadata.csv", index=False)
    print(f"Saved metadata: {out_dir / 'ccv2_metadata.csv'}")
    print(f"Videos: {len(df)}, subjects: {df['subject_id'].nunique()}")

    frames_root = Path(args.frames)
    rows = []
    for subject_dir in sorted(frames_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for f in sorted(subject_dir.glob("*.jpg")) + sorted(subject_dir.glob("*.png")):
            rows.append({"subject_id": subject_dir.name, "frame_path": str(f)})

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "frame_manifest.csv", index=False)
    print(f"Saved manifest: {out_dir / 'frame_manifest.csv'}")
    print(f"Total frames: {len(manifest)}, "
          f"subjects with frames: {manifest['subject_id'].nunique()}")


if __name__ == "__main__":
    main()
