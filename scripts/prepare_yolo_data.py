#!/usr/bin/env python3
"""Prepare the YOLO-style data directory from RetinaFace detections.

Converts RetinaFace bounding boxes to YOLO label files and creates
train/val splits with symlinks to the original frames.

Usage:
    python prepare_yolo_data.py \\
        --detections results/detections/retinaface_conf0.5_imgsz640.csv \\
        --frames /path/to/frames \\
        --out yolo_data \\
        --labels labels_retina
"""
import argparse
import os
import random
from pathlib import Path

import pandas as pd
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detections", required=True,
                   help="RetinaFace detections CSV")
    p.add_argument("--frames", required=True, help="Frames root folder")
    p.add_argument("--labels", required=True,
                   help="Output folder for YOLO .txt labels")
    p.add_argument("--out", required=True, help="Output folder for YOLO data")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    det = pd.read_csv(args.detections)
    det = det[det["detected"] == True].copy()
    print(f"Detected frames available: {len(det)}")

    labels_dir = Path(args.labels)
    labels_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    errors = 0
    for _, row in det.iterrows():
        try:
            img = Image.open(row["frame_path"])
            W, H = img.size
            img.close()

            x1, y1 = row["bbox_x1"], row["bbox_y1"]
            x2, y2 = row["bbox_x2"], row["bbox_y2"]
            xc = ((x1 + x2) / 2) / W
            yc = ((y1 + y2) / 2) / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H

            sid = Path(row["frame_path"]).parent.name
            fname = Path(row["frame_path"]).stem
            ldir = labels_dir / sid
            ldir.mkdir(exist_ok=True)
            with open(ldir / f"{fname}.txt", "a") as f:
                f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            count += 1
        except Exception:
            errors += 1

    print(f"Labels generated: {count}, errors: {errors}")

    frames_root = Path(args.frames)
    out_root = Path(args.out)
    subjects = sorted(d.name for d in labels_dir.iterdir() if d.is_dir())
    print(f"Subjects with labels: {len(subjects)}")

    random.seed(args.seed)
    random.shuffle(subjects)
    split = int(len(subjects) * (1 - args.val_fraction))
    splits = {"train": subjects[:split], "val": subjects[split:]}
    print(f"Train subjects: {len(splits['train'])}, "
          f"val subjects: {len(splits['val'])}")

    for split_name, subs in splits.items():
        img_dir = out_root / "images" / split_name
        lbl_dir = out_root / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        pairs = 0
        for sid in subs:
            fr = frames_root / sid
            lb = labels_dir / sid
            if not fr.exists() or not lb.exists():
                continue
            for lbl_file in lb.glob("*.txt"):
                stem = lbl_file.stem
                for ext in [".jpg", ".png"]:
                    img_file = fr / (stem + ext)
                    if img_file.exists():
                        link_name = f"{sid}_{stem}"
                        img_link = img_dir / (link_name + ext)
                        lbl_link = lbl_dir / (link_name + ".txt")
                        if not img_link.exists():
                            os.symlink(str(img_file), str(img_link))
                        if not lbl_link.exists():
                            os.symlink(str(lbl_file), str(lbl_link))
                        pairs += 1
                        break
        print(f"{split_name}: {pairs} image/label pairs")

    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_root.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        f.write("  0: face\n")
        f.write("nc: 1\n")
    print(f"Wrote data config: {yaml_path}")


if __name__ == "__main__":
    main()
