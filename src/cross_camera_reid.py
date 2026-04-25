"""
cross_camera_reid.py — Level 3: Cross-Camera Re-Identification

Given two video feeds from different camera viewpoints, runs the
multi-object tracker on each independently, then matches track
embeddings across cameras to assign consistent global IDs.

Usage:
    python cross_camera_reid.py --cam1 ..\Videos\cam1_clip1.mp4 \
                                 --cam2 ..\Videos\cam2_clip1.mp4 \
                                 --outdir ..\Results\cross_camera
"""

import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from kalman_filter import KalmanBoxTracker
from embedding_extractor import EmbeddingExtractor, cosine_distance
from multi_object_tracker import MultiObjectTracker, Track, draw_box


# ======================================================================
# Run tracker on a single video and collect track embeddings
# ======================================================================
def run_and_collect(video_path, embedder, lambda_weight=0.7, max_lost=50):
    """
    Run multi-object tracker on a video and return per-track gallery embeddings.

    Returns
    -------
    track_data : dict
        track_id -> {
            'gallery_mean': ndarray (512,),
            'gallery': list of ndarray (512,),
            'frames_seen': int,
            'first_frame': int,
            'last_frame': int,
        }
    frame_count : int
    """
    detector = YOLO("yolov8n.pt")
    Track.reset_counter()
    tracker = MultiObjectTracker(
        lambda_weight=lambda_weight,
        max_lost=max_lost,
        max_cost=0.7,
        min_hits=3,
        reid_threshold=0.4,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return {}, 0

    # Track data collection
    track_data = {}
    frame_idx = 0

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

        # Track
        track_results = tracker.update(person_dets, embeddings)

        # Collect embeddings from active tracks
        for trk in tracker.active_tracks:
            if trk.state == "confirmed":
                tid = trk.id
                if tid not in track_data:
                    track_data[tid] = {
                        'gallery': [],
                        'frames_seen': 0,
                        'first_frame': frame_idx,
                        'last_frame': frame_idx,
                    }
                track_data[tid]['gallery'].append(trk.embedding.copy())
                track_data[tid]['frames_seen'] += 1
                track_data[tid]['last_frame'] = frame_idx

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"    Frame {frame_idx}...")

    cap.release()

    # Compute gallery means
    for tid in track_data:
        gallery = np.array(track_data[tid]['gallery'])
        mean = gallery.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm > 1e-6:
            mean /= norm
        track_data[tid]['gallery_mean'] = mean
        # Keep only last 20 for storage
        track_data[tid]['gallery'] = track_data[tid]['gallery'][-20:]

    return track_data, frame_idx


# ======================================================================
# Cross-camera matching
# ======================================================================
def match_across_cameras(tracks_cam1, tracks_cam2, threshold=0.5):
    """
    Match tracks from Camera 1 to Camera 2 using gallery embedding similarity.

    Parameters
    ----------
    tracks_cam1 : dict from run_and_collect
    tracks_cam2 : dict from run_and_collect
    threshold   : float, max cosine distance to accept a match

    Returns
    -------
    matches : list of (cam1_id, cam2_id, distance)
    unmatched_cam1 : list of cam1_ids
    unmatched_cam2 : list of cam2_ids
    cost_matrix : ndarray
    """
    ids1 = [tid for tid in tracks_cam1 if tracks_cam1[tid]['frames_seen'] >= 10]
    ids2 = [tid for tid in tracks_cam2 if tracks_cam2[tid]['frames_seen'] >= 10]

    if len(ids1) == 0 or len(ids2) == 0:
        return [], list(tracks_cam1.keys()), list(tracks_cam2.keys()), np.array([])

    # Build cost matrix (cosine distance)
    cost = np.zeros((len(ids1), len(ids2)))
    for i, tid1 in enumerate(ids1):
        emb1 = tracks_cam1[tid1]['gallery_mean']
        for j, tid2 in enumerate(ids2):
            emb2 = tracks_cam2[tid2]['gallery_mean']
            cost[i, j] = cosine_distance(emb1, emb2)

    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    matched_i = set()
    matched_j = set()

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= threshold:
            matches.append((ids1[r], ids2[c], cost[r, c]))
            matched_i.add(r)
            matched_j.add(c)

    unmatched_cam1 = [ids1[i] for i in range(len(ids1)) if i not in matched_i]
    unmatched_cam2 = [ids2[j] for j in range(len(ids2)) if j not in matched_j]

    return matches, unmatched_cam1, unmatched_cam2, cost


# ======================================================================
# Generate annotated output videos with global IDs
# ======================================================================
def generate_global_id_video(video_path, output_path, embedder, global_id_map,
                              lambda_weight=0.7, max_lost=50):
    """Re-run tracker and write video with global IDs."""
    detector = YOLO("yolov8n.pt")
    Track.reset_counter()
    tracker = MultiObjectTracker(
        lambda_weight=lambda_weight, max_lost=max_lost,
        max_cost=0.7, min_hits=3, reid_threshold=0.4,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector(frame, device="cpu", verbose=False)[0]
        person_dets = []
        for box in results.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                person_dets.append(box.xyxy[0].cpu().numpy())

        crops = []
        for det in person_dets:
            bx1, by1, bx2, by2 = map(int, det)
            crop = frame[max(0, by1):max(0, by2), max(0, bx1):max(0, bx2)]
            crops.append(crop)
        embeddings = embedder.extract_batch(crops)

        track_results = tracker.update(person_dets, embeddings)

        for (tid, bbox, state) in track_results:
            if state == "confirmed":
                # Map local ID to global ID
                gid = global_id_map.get(tid, tid)
                draw_box(frame, bbox, gid, state)
                # Add "Global" label
                x1, y1 = int(bbox[0]), int(bbox[1])
                cv2.putText(frame, f"GID {gid}", (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        n_active = sum(1 for _, _, s in track_results if s == "confirmed")
        cv2.putText(frame, f"Frame {frame_idx}  Active: {n_active}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return frame_idx


# ======================================================================
# Visualization
# ======================================================================
def plot_distance_matrix(cost, ids1, ids2, matches, outdir):
    """Plot the cross-camera distance matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(cost, cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(ids2)))
    ax.set_xticklabels([f"Cam2 ID{t}" for t in ids2], fontsize=9, rotation=45)
    ax.set_yticks(range(len(ids1)))
    ax.set_yticklabels([f"Cam1 ID{t}" for t in ids1], fontsize=9)
    ax.set_xlabel("Camera 2 Tracks", fontsize=10)
    ax.set_ylabel("Camera 1 Tracks", fontsize=10)
    ax.set_title("Cross-Camera Cosine Distance Matrix", fontsize=12, fontweight="bold")

    # Annotate cells
    for i in range(cost.shape[0]):
        for j in range(cost.shape[1]):
            color = "white" if cost[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cost[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color=color)

    # Highlight matches
    for cam1_id, cam2_id, dist in matches:
        i = list(ids1).index(cam1_id) if cam1_id in ids1 else -1
        j = list(ids2).index(cam2_id) if cam2_id in ids2 else -1
        if i >= 0 and j >= 0:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=3,
                                  edgecolor="lime", facecolor="none")
            ax.add_patch(rect)

    plt.colorbar(im, ax=ax, label="Cosine Distance")
    plt.tight_layout()
    path = os.path.join(outdir, "cross_camera_distances.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved distance matrix → {path}")
    return path


def plot_matching_summary(matches, unmatched1, unmatched2, outdir):
    """Bar chart showing matched vs unmatched tracks."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    categories = ["Matched", "Unmatched\nCam 1", "Unmatched\nCam 2"]
    values = [len(matches), len(unmatched1), len(unmatched2)]
    colors = ["#27AE60", "#E74C3C", "#E74C3C"]
    ax.bar(categories, values, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Number of Tracks")
    ax.set_title("Cross-Camera Matching Results", fontsize=12, fontweight="bold")
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, str(v), ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(values) + 1.5)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, "matching_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved matching summary → {path}")
    return path


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Level 3: Cross-Camera Re-ID")
    parser.add_argument("--cam1", required=True, help="Camera 1 video path")
    parser.add_argument("--cam2", required=True, help="Camera 2 video path")
    parser.add_argument("--outdir", default="../Results/cross_camera", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Max cosine distance for cross-camera match")
    parser.add_argument("--lambda-weight", type=float, default=0.7)
    parser.add_argument("--max-lost", type=int, default=50)
    parser.add_argument("--save-video", action="store_true",
                        help="Generate output videos with global IDs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("[INFO] Loading embedding extractor ...")
    embedder = EmbeddingExtractor()

    # Step 1: Run tracker on Camera 1
    print(f"\n{'#'*60}")
    print(f"  Camera 1: {args.cam1}")
    print(f"{'#'*60}")
    tracks1, frames1 = run_and_collect(
        args.cam1, embedder, args.lambda_weight, args.max_lost
    )
    print(f"  Camera 1: {frames1} frames, {len(tracks1)} tracks")
    for tid, data in tracks1.items():
        print(f"    Track {tid}: {data['frames_seen']} frames "
              f"(frame {data['first_frame']}–{data['last_frame']})")

    # Step 2: Run tracker on Camera 2
    print(f"\n{'#'*60}")
    print(f"  Camera 2: {args.cam2}")
    print(f"{'#'*60}")
    tracks2, frames2 = run_and_collect(
        args.cam2, embedder, args.lambda_weight, args.max_lost
    )
    print(f"  Camera 2: {frames2} frames, {len(tracks2)} tracks")
    for tid, data in tracks2.items():
        print(f"    Track {tid}: {data['frames_seen']} frames "
              f"(frame {data['first_frame']}–{data['last_frame']})")

    # Step 3: Cross-camera matching
    print(f"\n{'='*60}")
    print(f"  Cross-Camera Matching (threshold={args.threshold})")
    print(f"{'='*60}")
    matches, unmatched1, unmatched2, cost = match_across_cameras(
        tracks1, tracks2, threshold=args.threshold
    )

    if len(matches) > 0:
        print(f"\n  Matches found: {len(matches)}")
        global_id = 1
        global_map_cam1 = {}
        global_map_cam2 = {}
        for cam1_id, cam2_id, dist in matches:
            print(f"    Cam1 Track {cam1_id} <-> Cam2 Track {cam2_id}  "
                  f"(cosine dist = {dist:.4f})  →  Global ID {global_id}")
            global_map_cam1[cam1_id] = global_id
            global_map_cam2[cam2_id] = global_id
            global_id += 1
    else:
        print("\n  No cross-camera matches found.")
        global_map_cam1 = {}
        global_map_cam2 = {}

    if len(unmatched1) > 0:
        print(f"\n  Unmatched Camera 1 tracks: {unmatched1}")
    if len(unmatched2) > 0:
        print(f"  Unmatched Camera 2 tracks: {unmatched2}")

    # Step 4: Visualizations
    if cost.size > 0:
        ids1 = [tid for tid in tracks1 if tracks1[tid]['frames_seen'] >= 10]
        ids2 = [tid for tid in tracks2 if tracks2[tid]['frames_seen'] >= 10]
        plot_distance_matrix(cost, ids1, ids2, matches, args.outdir)

    plot_matching_summary(matches, unmatched1, unmatched2, args.outdir)

    # Step 5: Generate global ID videos (optional)
    if args.save_video and len(matches) > 0:
        print("\n[INFO] Generating Camera 1 video with global IDs ...")
        cam1_out = os.path.join(args.outdir, "cam1_global_ids.mp4")
        generate_global_id_video(args.cam1, cam1_out, embedder,
                                  global_map_cam1, args.lambda_weight, args.max_lost)
        print(f"  Saved → {cam1_out}")

        print("[INFO] Generating Camera 2 video with global IDs ...")
        cam2_out = os.path.join(args.outdir, "cam2_global_ids.mp4")
        generate_global_id_video(args.cam2, cam2_out, embedder,
                                  global_map_cam2, args.lambda_weight, args.max_lost)
        print(f"  Saved → {cam2_out}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  CROSS-CAMERA RE-ID SUMMARY")
    print(f"{'='*60}")
    print(f"  Camera 1: {len(tracks1)} tracks, {frames1} frames")
    print(f"  Camera 2: {len(tracks2)} tracks, {frames2} frames")
    print(f"  Matches : {len(matches)}")
    print(f"  Unmatched Cam1: {len(unmatched1)}")
    print(f"  Unmatched Cam2: {len(unmatched2)}")
    if len(matches) > 0:
        avg_dist = np.mean([d for _, _, d in matches])
        print(f"  Avg match distance: {avg_dist:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
