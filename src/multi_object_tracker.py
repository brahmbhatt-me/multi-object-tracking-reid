"""
multi_object_tracker.py — Level 2: Multi-Object Tracking with Re-ID

Extends the Level 1 single-object tracker to handle multiple simultaneous
objects with identity recovery after occlusion using deep embeddings.

Features:
  - Track pool with lifecycle management (tentative → confirmed → lost → deleted)
  - Hungarian algorithm for optimal detection-to-track assignment
  - Combined cost matrix: λ·(1−IoU) + (1−λ)·cosine_distance
  - Re-ID gallery: match re-entering objects to stored embeddings
  - Exponential moving average embedding updates

Usage:
    python multi_object_tracker.py --video path/to/video.mp4 \
                                    --output results/output.mp4 \
                                    [--lambda-weight 0.5] [--max-lost 30]
"""

import argparse
import os
import sys
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from kalman_filter import KalmanBoxTracker
from embedding_extractor import EmbeddingExtractor, cosine_distance


# ======================================================================
# Utility: IoU matrix
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


def iou_matrix(tracks, detections):
    """Compute IoU matrix between all tracks and detections."""
    n_tracks = len(tracks)
    n_dets = len(detections)
    matrix = np.zeros((n_tracks, n_dets))
    for i, trk in enumerate(tracks):
        for j, det in enumerate(detections):
            matrix[i, j] = iou(trk, det)
    return matrix


def embedding_distance_matrix(track_embs, det_embs):
    """Compute cosine distance matrix between track and detection embeddings."""
    n_tracks = len(track_embs)
    n_dets = len(det_embs)
    matrix = np.zeros((n_tracks, n_dets))
    for i in range(n_tracks):
        for j in range(n_dets):
            matrix[i, j] = cosine_distance(track_embs[i], det_embs[j])
    return matrix


# ======================================================================
# Track class with embedding gallery
# ======================================================================
class Track:
    """A tracked object with Kalman state and appearance embedding."""

    _next_id = 1

    def __init__(self, bbox, embedding, min_hits=3):
        self.kalman = KalmanBoxTracker.__new__(KalmanBoxTracker)
        # Manually init without using class counter
        self.id = Track._next_id
        Track._next_id += 1

        # Init Kalman state
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.kalman.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)
        self.kalman.F = np.eye(8)
        self.kalman.F[0, 4] = 1.0
        self.kalman.F[1, 5] = 1.0
        self.kalman.F[2, 6] = 1.0
        self.kalman.F[3, 7] = 1.0
        self.kalman.H = np.zeros((4, 8))
        for i in range(4):
            self.kalman.H[i, i] = 1
        self.kalman.P = np.eye(8) * 10.0
        self.kalman.P[4:, 4:] *= 100.0
        self.kalman.Q = np.eye(8) * 1.0
        self.kalman.Q[4:, 4:] *= 0.01
        self.kalman.R = np.eye(4) * 1.0
        self.kalman.time_since_update = 0
        self.kalman.hits = 1
        self.kalman.age = 0

        # Appearance
        self.embedding = embedding.copy()
        self.gallery = [embedding.copy()]  # Store embeddings for Re-ID

        # Lifecycle
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.min_hits = min_hits
        self.state = "tentative"  # tentative → confirmed → lost → deleted

    def predict(self):
        """Advance Kalman state and return predicted bbox."""
        pred = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        return pred

    def update(self, bbox, embedding):
        """Update track with matched detection."""
        self.kalman.update(bbox)
        self.time_since_update = 0
        self.hits += 1

        # Update embedding with EMA
        self.embedding = 0.9 * self.embedding + 0.1 * embedding
        norm = np.linalg.norm(self.embedding)
        if norm > 1e-6:
            self.embedding /= norm

        # Store in gallery (keep last 10)
        self.gallery.append(embedding.copy())
        if len(self.gallery) > 10:
            self.gallery.pop(0)

        # Update lifecycle state
        if self.state == "tentative" and self.hits >= self.min_hits:
            self.state = "confirmed"
        elif self.state == "lost":
            self.state = "confirmed"

    def mark_lost(self):
        """Mark track as lost."""
        if self.state == "confirmed":
            self.state = "lost"

    def get_state(self):
        """Return current bbox as [x1, y1, x2, y2]."""
        return self.kalman.get_state()

    def get_gallery_mean(self):
        """Return mean embedding from gallery for Re-ID matching."""
        if len(self.gallery) == 0:
            return self.embedding
        arr = np.array(self.gallery)
        mean = arr.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm > 1e-6:
            mean /= norm
        return mean

    @staticmethod
    def reset_counter():
        Track._next_id = 1


# ======================================================================
# Drawing helpers
# ======================================================================
_COLORS = [
    (46, 204, 113), (52, 152, 219), (231, 76, 60),
    (241, 196, 15), (155, 89, 182), (26, 188, 156),
    (230, 126, 34), (149, 165, 166), (192, 57, 43),
    (41, 128, 185), (39, 174, 96), (142, 68, 173),
]


def draw_box(frame, box, track_id, state="confirmed"):
    """Draw bounding box with ID label."""
    color = _COLORS[track_id % len(_COLORS)]
    if state == "lost":
        color = (0, 0, 200)
    elif state == "tentative":
        color = (180, 180, 180)

    x1, y1, x2, y2 = map(int, box)
    thickness = 2 if state == "confirmed" else 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label = f"ID {track_id}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ======================================================================
# Multi-Object Tracker
# ======================================================================
class MultiObjectTracker:
    """
    Multi-object tracker with Re-ID capability.

    Parameters
    ----------
    lambda_weight : float
        Balance between IoU and embedding distance in cost matrix.
        0 = pure embedding, 1 = pure IoU.
    max_lost : int
        Frames before a lost track is deleted.
    max_cost : float
        Maximum cost to accept an assignment.
    min_hits : int
        Consecutive detections before a track is confirmed.
    reid_threshold : float
        Maximum embedding distance to accept a Re-ID match.
    """

    def __init__(self, lambda_weight=0.5, max_lost=30, max_cost=0.7,
                 min_hits=3, reid_threshold=0.4):
        self.lambda_w = lambda_weight
        self.max_lost = max_lost
        self.max_cost = max_cost
        self.min_hits = min_hits
        self.reid_threshold = reid_threshold

        self.active_tracks = []
        self.lost_tracks = []

    def update(self, detections, embeddings):
        """
        Run one tracking step.

        Parameters
        ----------
        detections : list of ndarray, each (4,) [x1, y1, x2, y2]
        embeddings : ndarray, shape (N, 512)

        Returns
        -------
        results : list of (track_id, bbox, state)
        """
        # 1. Predict all active tracks
        predicted_boxes = []
        for trk in self.active_tracks:
            pred = trk.predict()
            predicted_boxes.append(pred)

        # 2. Associate detections to active tracks (Hungarian algorithm)
        matched, unmatched_dets, unmatched_trks = self._associate(
            predicted_boxes, detections, embeddings
        )

        # 3. Update matched tracks
        for trk_idx, det_idx in matched:
            self.active_tracks[trk_idx].update(
                detections[det_idx], embeddings[det_idx]
            )

        # 4. Handle unmatched tracks → mark lost
        for trk_idx in unmatched_trks:
            self.active_tracks[trk_idx].mark_lost()

        # 5. Try Re-ID for unmatched detections against lost tracks
        still_unmatched_dets = []
        for det_idx in unmatched_dets:
            reid_match = self._try_reid(detections[det_idx], embeddings[det_idx])
            if reid_match is not None:
                # Recover lost track
                reid_match.update(detections[det_idx], embeddings[det_idx])
                self.active_tracks.append(reid_match)
                self.lost_tracks.remove(reid_match)
            else:
                still_unmatched_dets.append(det_idx)

        # 6. Create new tracks for remaining unmatched detections
        for det_idx in still_unmatched_dets:
            new_track = Track(detections[det_idx], embeddings[det_idx],
                              min_hits=self.min_hits)
            self.active_tracks.append(new_track)

        # 7. Move lost tracks, delete old ones
        tracks_to_keep = []
        for trk in self.active_tracks:
            if trk.state == "lost":
                self.lost_tracks.append(trk)
            else:
                tracks_to_keep.append(trk)
        self.active_tracks = tracks_to_keep

        # Prune lost tracks that exceeded max_lost
        self.lost_tracks = [
            t for t in self.lost_tracks
            if t.time_since_update <= self.max_lost
        ]

        # 8. Collect results (only confirmed tracks)
        results = []
        for trk in self.active_tracks:
            if trk.state == "confirmed" or (trk.state == "tentative" and trk.hits >= 1):
                bbox = trk.get_state()
                results.append((trk.id, bbox, trk.state))

        return results

    def _associate(self, pred_boxes, detections, embeddings):
        """
        Associate predictions to detections using Hungarian algorithm
        with combined IoU + embedding cost.
        """
        if len(pred_boxes) == 0 or len(detections) == 0:
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(pred_boxes)))
            return [], unmatched_dets, unmatched_trks

        # Compute IoU matrix
        iou_mat = iou_matrix(pred_boxes, detections)
        iou_cost = 1.0 - iou_mat  # IoU distance

        # Compute embedding distance matrix
        track_embs = [trk.embedding for trk in self.active_tracks]
        emb_cost = embedding_distance_matrix(track_embs, embeddings)

        # Combined cost
        cost = self.lambda_w * iou_cost + (1.0 - self.lambda_w) * emb_cost

        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost)

        matched = []
        unmatched_dets = set(range(len(detections)))
        unmatched_trks = set(range(len(pred_boxes)))

        for r, c in zip(row_indices, col_indices):
            if cost[r, c] > self.max_cost:
                continue
            matched.append((r, c))
            unmatched_dets.discard(c)
            unmatched_trks.discard(r)

        return matched, list(unmatched_dets), list(unmatched_trks)

    def _try_reid(self, detection, embedding):
        """
        Try to match a detection to a lost track via embedding similarity.

        Returns the matched Track or None.
        """
        if len(self.lost_tracks) == 0:
            return None

        best_match = None
        best_dist = self.reid_threshold

        for trk in self.lost_tracks:
            gallery_emb = trk.get_gallery_mean()
            dist = cosine_distance(gallery_emb, embedding)
            if dist < best_dist:
                best_dist = dist
                best_match = trk

        return best_match


# ======================================================================
# Main pipeline
# ======================================================================
def run_multi_object_tracker(
    video_path,
    output_path=None,
    lambda_weight=0.5,
    max_lost=30,
    max_cost=0.7,
    min_hits=3,
    reid_threshold=0.4,
    show=True,
):
    """Run the Level 2 multi-object tracker."""
    print("[INFO] Loading YOLOv8n detector …")
    detector = YOLO("yolov8n.pt")

    print("[INFO] Loading ResNet18 embedding extractor …")
    embedder = EmbeddingExtractor()

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

    Track.reset_counter()
    tracker = MultiObjectTracker(
        lambda_weight=lambda_weight,
        max_lost=max_lost,
        max_cost=max_cost,
        min_hits=min_hits,
        reid_threshold=reid_threshold,
    )

    frame_idx = 0
    total_ids = set()
    id_switches = 0
    prev_ids = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        # Update tracker
        track_results = tracker.update(person_dets, embeddings)

        # Draw results
        for (tid, bbox, state) in track_results:
            total_ids.add(tid)
            if state == "confirmed":
                draw_box(frame, bbox, tid, state)

        # Status overlay
        n_active = sum(1 for _, _, s in track_results if s == "confirmed")
        n_lost = len(tracker.lost_tracks)
        cv2.putText(frame, f"Frame {frame_idx}/{total}  Active: {n_active}  Lost: {n_lost}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if writer:
            writer.write(frame)
        if show:
            cv2.imshow("Multi-Object Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\n{'='*55}")
    print(f"  MOT Summary — {os.path.basename(video_path)}")
    print(f"{'='*55}")
    print(f"  Frames processed   : {frame_idx}")
    print(f"  Unique IDs assigned : {len(total_ids)}")
    print(f"  Lost tracks (final) : {len(tracker.lost_tracks)}")
    print(f"  Lambda (IoU weight) : {lambda_weight}")
    print(f"  Re-ID threshold     : {reid_threshold}")
    print(f"{'='*55}\n")

    return frame_idx, total_ids


# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Level 2: Multi-Object Tracker with Re-ID"
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default=None, help="Path to save annotated video")
    parser.add_argument("--lambda-weight", type=float, default=0.5,
                        help="Balance: 0=pure embedding, 1=pure IoU (default 0.5)")
    parser.add_argument("--max-lost", type=int, default=30,
                        help="Frames before a lost track is deleted")
    parser.add_argument("--max-cost", type=float, default=0.7,
                        help="Maximum cost to accept an assignment")
    parser.add_argument("--min-hits", type=int, default=3,
                        help="Consecutive hits before a track is confirmed")
    parser.add_argument("--reid-thresh", type=float, default=0.4,
                        help="Maximum embedding distance for Re-ID match")
    parser.add_argument("--no-show", action="store_true",
                        help="Disable live display window")
    args = parser.parse_args()

    run_multi_object_tracker(
        video_path=args.video,
        output_path=args.output,
        lambda_weight=args.lambda_weight,
        max_lost=args.max_lost,
        max_cost=args.max_cost,
        min_hits=args.min_hits,
        reid_threshold=args.reid_thresh,
        show=not args.no_show,
    )
