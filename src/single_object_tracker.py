"""
single_object_tracker.py — Level 1: Single-Object Tracking Pipeline

Reads a video, runs YOLOv8 detection on each frame, and tracks a single
user-selected object across frames using a Kalman filter.  Also extracts
a 512-d ResNet18 embedding per detection for later use in Level 2 Re-ID.

Usage:
    python single_object_tracker.py --video path/to/video.mp4 \
                                     --output results/output.mp4 \
                                     [--select-id 0]
"""

import argparse
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Local modules
from kalman_filter import KalmanBoxTracker
from embedding_extractor import EmbeddingExtractor


# ======================================================================
# Utility: IoU
# ======================================================================
def iou(box_a, box_b):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ======================================================================
# Association: nearest detection to tracked box by IoU
# ======================================================================
def best_detection_match(predicted_box, detections, iou_threshold=0.3):
    """
    Find the detection with the highest IoU to the predicted box.

    Parameters
    ----------
    predicted_box : array (4,)  [x1, y1, x2, y2]
    detections    : list of arrays, each (4,)
    iou_threshold : float

    Returns
    -------
    best_idx : int or None
    best_iou : float
    """
    best_idx = None
    best_iou = 0.0
    for i, det in enumerate(detections):
        score = iou(predicted_box, det)
        if score > best_iou:
            best_iou = score
            best_idx = i
    if best_iou < iou_threshold:
        return None, best_iou
    return best_idx, best_iou


# ======================================================================
# Drawing helpers
# ======================================================================
_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
]


def draw_box(frame, box, track_id, color=None):
    """Draw a bounding box with a persistent ID label."""
    if color is None:
        color = _COLORS[track_id % len(_COLORS)]
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {track_id}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# ======================================================================
# Main tracking loop
# ======================================================================
def run_single_object_tracker(
    video_path,
    output_path=None,
    select_id=0,
    max_lost_frames=30,
    iou_threshold=0.3,
    show=True,
):
    """
    Run the Level 1 single-object Kalman filter tracker.

    Parameters
    ----------
    video_path       : str   – path to input video
    output_path      : str   – path to save annotated output video (optional)
    select_id        : int   – index of the detection to track in the first frame
    max_lost_frames  : int   – frames before the track is declared lost
    iou_threshold    : float – minimum IoU to accept a match
    show             : bool  – display live window

    Returns
    -------
    iou_scores : list of float – per-frame IoU between prediction and matched detection
    """
    # ---- Load models ----
    print("[INFO] Loading YOLOv8n detector …")
    detector = YOLO("yolov8n.pt")

    print("[INFO] Loading ResNet18 embedding extractor …")
    embedder = EmbeddingExtractor()

    # ---- Open video ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # ---- First frame: initialise tracker ----
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read the first frame.")
        sys.exit(1)

    results = detector(frame, device="cpu", verbose=False)[0]
    person_dets = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # COCO class 0 = person
            xyxy = box.xyxy[0].cpu().numpy()
            person_dets.append(xyxy)

    if len(person_dets) == 0:
        print("[ERROR] No person detected in the first frame.")
        sys.exit(1)

    select_id = min(select_id, len(person_dets) - 1)
    init_box = person_dets[select_id]

    KalmanBoxTracker.reset_count()
    tracker = KalmanBoxTracker(init_box)
    print(f"[INFO] Initialised tracker (ID {tracker.id}) on detection {select_id}")

    # Extract initial embedding
    x1, y1, x2, y2 = map(int, init_box)
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    gallery_emb = embedder.extract(crop)
    print(f"[INFO] Initial embedding norm = {np.linalg.norm(gallery_emb):.4f}")

    # Draw first frame
    draw_box(frame, init_box, tracker.id)
    if writer:
        writer.write(frame)

    # ---- Tracking loop ----
    iou_scores = []
    frame_idx = 1
    lost_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Predict
        pred_box = tracker.predict()

        # 2. Detect persons
        results = detector(frame, device="cpu", verbose=False)[0]
        person_dets = []
        for box in results.boxes:
            if int(box.cls[0]) == 0:
                person_dets.append(box.xyxy[0].cpu().numpy())

        # 3. Associate: find detection with highest IoU to prediction
        match_idx, match_iou = best_detection_match(
            pred_box, person_dets, iou_threshold
        )

        if match_idx is not None:
            matched_box = person_dets[match_idx]
            tracker.update(matched_box)
            iou_scores.append(match_iou)
            lost_count = 0

            # Update gallery embedding (exponential moving average)
            bx1, by1, bx2, by2 = map(int, matched_box)
            crop = frame[max(0, by1):max(0, by2), max(0, bx1):max(0, bx2)]
            new_emb = embedder.extract(crop)
            gallery_emb = 0.9 * gallery_emb + 0.1 * new_emb
            norm = np.linalg.norm(gallery_emb)
            if norm > 1e-6:
                gallery_emb /= norm

            display_box = tracker.get_state()
        else:
            # No match — use prediction only
            iou_scores.append(0.0)
            lost_count += 1
            display_box = pred_box

        # Draw
        colour = (0, 255, 0) if lost_count == 0 else (0, 0, 255)
        draw_box(frame, display_box, tracker.id, colour)

        # Status overlay
        status = "TRACKING" if lost_count == 0 else f"LOST ({lost_count})"
        cv2.putText(frame, f"Frame {frame_idx}/{total}  {status}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if writer:
            writer.write(frame)
        if show:
            cv2.imshow("Single-Object Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if lost_count >= max_lost_frames:
            print(f"[WARN] Track lost for {max_lost_frames} frames — stopping.")
            break

    # ---- Clean up ----
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    # ---- Summary ----
    valid = [s for s in iou_scores if s > 0]
    mean_iou = np.mean(valid) if valid else 0.0
    print(f"\n{'='*50}")
    print(f"  Tracking Summary  —  {os.path.basename(video_path)}")
    print(f"{'='*50}")
    print(f"  Total frames processed : {frame_idx}")
    print(f"  Frames with match      : {len(valid)}")
    print(f"  Frames lost            : {frame_idx - len(valid)}")
    print(f"  Mean IoU (matched)     : {mean_iou:.4f}")
    print(f"  Track ID               : {tracker.id}")
    print(f"{'='*50}\n")

    return iou_scores


# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Level 1: Single-Object Kalman Filter Tracker with YOLOv8"
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default=None, help="Path to save annotated video")
    parser.add_argument("--select-id", type=int, default=0,
                        help="Index of person detection to track in frame 0")
    parser.add_argument("--max-lost", type=int, default=30,
                        help="Max consecutive lost frames before stopping")
    parser.add_argument("--iou-thresh", type=float, default=0.3,
                        help="Minimum IoU to accept a match")
    parser.add_argument("--no-show", action="store_true",
                        help="Disable live display window")
    args = parser.parse_args()

    scores = run_single_object_tracker(
        video_path=args.video,
        output_path=args.output,
        select_id=args.select_id,
        max_lost_frames=args.max_lost,
        iou_threshold=args.iou_thresh,
        show=not args.no_show,
    )