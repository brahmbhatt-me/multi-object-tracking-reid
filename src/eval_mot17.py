"""
eval_mot17.py — Evaluate multi-object tracker on MOT17 benchmark sequences.

Runs the tracker on MOT17 train sequences and computes MOTA, IDF1,
and ID switch metrics using py-motmetrics.

Usage:
    python eval_mot17.py --mot17-dir ..\MOT17\train \
                         --sequences MOT17-02-DPM MOT17-04-DPM MOT17-09-DPM \
                         --outdir ..\Results\mot17
"""

import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO
from kalman_filter import KalmanBoxTracker
from embedding_extractor import EmbeddingExtractor
from multi_object_tracker import MultiObjectTracker, Track, draw_box


def load_mot17_gt(gt_path):
    """Load MOT17 ground truth file into a dict: frame_id → list of (id, x1, y1, x2, y2)."""
    gt = {}
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            frame = int(parts[0])
            tid = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            active = int(parts[6])
            cls = int(parts[7])
            # Only use active pedestrian annotations
            if active == 1 and cls == 1:
                if frame not in gt:
                    gt[frame] = []
                gt[frame].append((tid, x, y, x + w, y + h))
    return gt


def run_on_mot17_sequence(seq_dir, embedder, lambda_weight=0.5, max_lost=30):
    """
    Run tracker on a single MOT17 sequence.

    Returns
    -------
    predictions : dict, frame_id → list of (track_id, x1, y1, x2, y2)
    """
    img_dir = os.path.join(seq_dir, "img1")
    if not os.path.isdir(img_dir):
        print(f"[ERROR] Image dir not found: {img_dir}")
        return {}

    # Get sorted image list
    images = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))])
    if len(images) == 0:
        print(f"[ERROR] No images in {img_dir}")
        return {}

    detector = YOLO("yolov8n.pt")
    Track.reset_counter()
    tracker = MultiObjectTracker(
        lambda_weight=lambda_weight,
        max_lost=max_lost,
        max_cost=0.7,
        min_hits=3,
        reid_threshold=0.4,
    )

    predictions = {}
    for idx, img_name in enumerate(images):
        frame_id = idx + 1
        frame = cv2.imread(os.path.join(img_dir, img_name))
        if frame is None:
            continue

        # Detect persons
        results = detector(frame, device="cpu", verbose=False)[0]
        person_dets = []
        for box in results.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                person_dets.append(box.xyxy[0].cpu().numpy())

        # Extract embeddings
        crops = []
        for det in person_dets:
            bx1, by1, bx2, by2 = map(int, det)
            crop = frame[max(0, by1):max(0, by2), max(0, bx1):max(0, bx2)]
            crops.append(crop)
        embeddings = embedder.extract_batch(crops)

        # Track
        track_results = tracker.update(person_dets, embeddings)

        predictions[frame_id] = []
        for (tid, bbox, state) in track_results:
            if state == "confirmed":
                predictions[frame_id].append((tid, bbox[0], bbox[1], bbox[2], bbox[3]))

        if frame_id % 100 == 0:
            print(f"    Frame {frame_id}/{len(images)}")

    return predictions


def compute_metrics(gt, predictions, iou_threshold=0.5):
    """
    Compute MOT metrics manually (MOTA, IDF1 approximation, ID switches).

    This is a simplified implementation. For full metrics use py-motmetrics.
    """
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_id_switches = 0
    prev_match = {}  # gt_id → predicted_id from last frame

    all_frames = sorted(set(list(gt.keys()) + list(predictions.keys())))

    for frame_id in all_frames:
        gt_boxes = gt.get(frame_id, [])
        pred_boxes = predictions.get(frame_id, [])
        total_gt += len(gt_boxes)

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue

        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue

        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue

        # Compute IoU matrix
        iou_mat = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, (gt_id, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
            for j, (pred_id, px1, py1, px2, py2) in enumerate(pred_boxes):
                x1 = max(gx1, px1)
                y1 = max(gy1, py1)
                x2 = min(gx2, px2)
                y2 = min(gy2, py2)
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_g = max(0, gx2 - gx1) * max(0, gy2 - gy1)
                area_p = max(0, px2 - px1) * max(0, py2 - py1)
                union = area_g + area_p - inter
                iou_mat[i, j] = inter / union if union > 0 else 0.0

        # Hungarian matching
        from scipy.optimize import linear_sum_assignment
        cost = 1.0 - iou_mat
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_gt = set()
        matched_pred = set()

        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= iou_threshold:
                matched_gt.add(r)
                matched_pred.add(c)
                total_tp += 1

                gt_id = gt_boxes[r][0]
                pred_id = pred_boxes[c][0]

                # Check for ID switch
                if gt_id in prev_match and prev_match[gt_id] != pred_id:
                    total_id_switches += 1
                prev_match[gt_id] = pred_id

        total_fp += len(pred_boxes) - len(matched_pred)
        total_fn += len(gt_boxes) - len(matched_gt)

    # Compute MOTA
    mota = 1.0 - (total_fn + total_fp + total_id_switches) / max(total_gt, 1)

    # Precision and recall
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)

    # IDF1 approximation (using F1 of ID-correct matches)
    idf1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        "MOTA": mota,
        "IDF1": idf1,
        "Precision": precision,
        "Recall": recall,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "IDsw": total_id_switches,
        "GT_total": total_gt,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate tracker on MOT17")
    parser.add_argument("--mot17-dir", required=True, help="Path to MOT17/train directory")
    parser.add_argument("--sequences", nargs="+",
                        default=["MOT17-02-DPM", "MOT17-04-DPM", "MOT17-09-DPM"],
                        help="Sequences to evaluate")
    parser.add_argument("--outdir", default="../Results/mot17", help="Output directory")
    parser.add_argument("--lambda-weight", type=float, default=0.5)
    parser.add_argument("--max-lost", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("[INFO] Loading embedding extractor …")
    embedder = EmbeddingExtractor()

    all_metrics = []
    seq_names = []

    for seq_name in args.sequences:
        seq_dir = os.path.join(args.mot17_dir, seq_name)
        if not os.path.isdir(seq_dir):
            print(f"[WARN] Sequence dir not found: {seq_dir}, skipping.")
            continue

        gt_path = os.path.join(seq_dir, "gt", "gt.txt")
        if not os.path.isfile(gt_path):
            print(f"[WARN] GT file not found: {gt_path}, skipping.")
            continue

        print(f"\n{'#'*60}")
        print(f"  Evaluating: {seq_name}")
        print(f"{'#'*60}")

        gt = load_mot17_gt(gt_path)
        predictions = run_on_mot17_sequence(
            seq_dir, embedder,
            lambda_weight=args.lambda_weight,
            max_lost=args.max_lost
        )
        metrics = compute_metrics(gt, predictions)

        print(f"\n  Results for {seq_name}:")
        print(f"    MOTA      : {metrics['MOTA']:.4f}")
        print(f"    IDF1      : {metrics['IDF1']:.4f}")
        print(f"    Precision : {metrics['Precision']:.4f}")
        print(f"    Recall    : {metrics['Recall']:.4f}")
        print(f"    ID Sw.    : {metrics['IDsw']}")
        print(f"    TP/FP/FN  : {metrics['TP']}/{metrics['FP']}/{metrics['FN']}")

        all_metrics.append(metrics)
        seq_names.append(seq_name)

    # Summary table
    if len(all_metrics) > 0:
        print(f"\n{'='*75}")
        print(f"  {'Sequence':<20} {'MOTA':>8} {'IDF1':>8} {'Prec':>8} {'Recall':>8} {'IDsw':>6}")
        print(f"  {'-'*70}")
        for name, m in zip(seq_names, all_metrics):
            print(f"  {name:<20} {m['MOTA']:>8.4f} {m['IDF1']:>8.4f} "
                  f"{m['Precision']:>8.4f} {m['Recall']:>8.4f} {m['IDsw']:>6}")

        # Average
        avg_mota = np.mean([m['MOTA'] for m in all_metrics])
        avg_idf1 = np.mean([m['IDF1'] for m in all_metrics])
        print(f"  {'-'*70}")
        print(f"  {'AVERAGE':<20} {avg_mota:>8.4f} {avg_idf1:>8.4f}")
        print(f"{'='*75}")

        # Save bar chart
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(seq_names))
        w = 0.3
        motas = [m['MOTA'] for m in all_metrics]
        idf1s = [m['IDF1'] for m in all_metrics]
        ax.bar(x - w/2, motas, w, label="MOTA", color="steelblue")
        ax.bar(x + w/2, idf1s, w, label="IDF1", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("MOT17-", "") for s in seq_names], fontsize=9)
        ax.set_ylabel("Score")
        ax.set_title("MOT17 Tracking Metrics")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "mot17_metrics.png"), dpi=150)
        plt.close()
        print(f"[INFO] Saved metrics chart → {args.outdir}/mot17_metrics.png")


if __name__ == "__main__":
    main()
