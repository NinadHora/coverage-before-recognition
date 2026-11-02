#!/usr/bin/env python3
"""Unified face detection on the CCv2 frame manifest.

Runs one detector against every frame listed in the manifest and writes a CSV
with one row per frame recording: frame_path, subject_id, n_detections,
max_conf, bounding box coordinates, detected (bool).

Supported detectors:
    haar, mtcnn, retinaface, scrfd, yolo11_ft, yolo12, rtdetr

Usage:
    python 01_run_detection.py --detector retinaface --conf 0.5
    python 01_run_detection.py --detector yolo11_ft --conf 0.25 \\
        --weights /path/to/custom.pt --output-name yolo11_scratch
"""
import argparse
import csv
import os
import time

import cv2
import pandas as pd
import torch
from tqdm import tqdm

MANIFEST = "configs/frame_manifest.csv"
RESULTS_DIR = "results/detections"


class HaarDetector:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, img, conf=0.0):
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        return [
            {"x1": int(x), "y1": int(y),
             "x2": int(x + w), "y2": int(y + h), "conf": 1.0}
            for x, y, w, h in faces
        ]


class MTCNNDetector:
    def __init__(self):
        from facenet_pytorch import MTCNN
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m = MTCNN(
            thresholds=[0.6, 0.7, 0.7], min_face_size=20,
            keep_all=True, device=self.dev,
        )

    def detect(self, img, conf=0.0):
        boxes, probs = self.m.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if boxes is None:
            return []
        return [
            {"x1": int(b[0]), "y1": int(b[1]),
             "x2": int(b[2]), "y2": int(b[3]), "conf": float(p)}
            for b, p in zip(boxes, probs) if p >= conf
        ]


class InsightFaceDetector:
    def __init__(self, model_name="buffalo_l"):
        import insightface
        self.app = insightface.app.FaceAnalysis(
            name=model_name,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, img, conf=0.5):
        faces = self.app.get(img)
        return [
            {"x1": int(f.bbox[0]), "y1": int(f.bbox[1]),
             "x2": int(f.bbox[2]), "y2": int(f.bbox[3]),
             "conf": float(f.det_score)}
            for f in faces if f.det_score >= conf
        ]


class UltralyticsDetector:
    def __init__(self, weights, imgsz=640, filter_class=None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.fc = filter_class

    def detect(self, img, conf=0.25):
        res = self.model(img, imgsz=self.imgsz, conf=conf, verbose=False)[0]
        dets = []
        for box in res.boxes:
            cid = int(box.cls[0])
            if self.fc is not None and cid != self.fc:
                continue
            xy = box.xyxy[0].cpu().numpy()
            dets.append({
                "x1": int(xy[0]), "y1": int(xy[1]),
                "x2": int(xy[2]), "y2": int(xy[3]),
                "conf": float(box.conf[0]),
            })
        return dets


def build(name, imgsz=640, weights=None):
    if name == "haar":
        return HaarDetector()
    if name == "mtcnn":
        return MTCNNDetector()
    if name == "retinaface":
        return InsightFaceDetector("buffalo_l")
    if name == "scrfd":
        return InsightFaceDetector("buffalo_sc")
    if name == "yolo11_ft":
        return UltralyticsDetector(weights or "yolo11n.pt", imgsz)
    if name == "yolo12":
        return UltralyticsDetector(weights or "yolo12n.pt", imgsz)
    if name == "rtdetr":
        return UltralyticsDetector(
            weights or "rtdetr-l.pt", imgsz, filter_class=0)
    raise ValueError(f"Unknown detector: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--detector", required=True,
        choices=["haar", "mtcnn", "retinaface", "scrfd",
                 "yolo11_ft", "yolo12", "rtdetr"],
    )
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--weights", type=str, default=None,
                   help="Path to custom weights")
    p.add_argument("--output-name", type=str, default=None,
                   help="Custom output filename stem")
    p.add_argument("--manifest", type=str, default=MANIFEST)
    p.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    out_name = args.output_name or args.detector
    out_path = os.path.join(
        args.results_dir,
        f"{out_name}_conf{args.conf}_imgsz{args.imgsz}.csv",
    )

    df = pd.read_csv(args.manifest)
    if args.max_frames:
        df = df.head(args.max_frames)

    gpu = (torch.cuda.get_device_name(0)
           if torch.cuda.is_available() else "CPU")
    print("=" * 60)
    print(f" Detection run: {out_name}")
    print(f" Confidence threshold: {args.conf}")
    print(f" Input size: {args.imgsz}")
    print(f" Frames: {len(df)}")
    print(f" Device: {gpu}")
    if args.weights:
        print(f" Weights: {args.weights}")
    print("=" * 60)

    det = build(args.detector, args.imgsz, args.weights)
    t0 = time.time()
    errors = 0

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_path", "subject_id", "n_detections", "max_conf",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "detected",
        ])
        for _, row in tqdm(df.iterrows(), total=len(df), desc=out_name):
            try:
                img = cv2.imread(row["frame_path"])
                if img is None:
                    w.writerow([row["frame_path"], row["subject_id"],
                                0, "", "", "", "", "", False])
                    continue
                dets = det.detect(img, conf=args.conf)
                if len(dets) == 0:
                    w.writerow([row["frame_path"], row["subject_id"],
                                0, "", "", "", "", "", False])
                else:
                    best = max(dets, key=lambda d: d["conf"])
                    w.writerow([
                        row["frame_path"], row["subject_id"],
                        len(dets), best["conf"],
                        best["x1"], best["y1"], best["x2"], best["y2"],
                        True,
                    ])
            except Exception:
                w.writerow([row["frame_path"], row["subject_id"],
                            0, "", "", "", "", "", False])
                errors += 1

    elapsed = (time.time() - t0) / 60
    print(f"\nDone: {out_path}")
    print(f"Time: {elapsed:.1f} min, Errors: {errors}")


if __name__ == "__main__":
    main()
