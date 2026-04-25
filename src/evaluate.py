"""
evaluate.py — Evaluate single-object tracker on multiple test clips.

Runs the tracker on each video in a directory (or a list of paths),
records per-frame IoU, and produces summary plots.

Usage:
    python evaluate.py --videos videos/clip1.mp4 videos/clip2.mp4 videos/clip3.mp4 \
                       --outdir results/
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_object_tracker import run_single_object_tracker


def plot_iou_curves(all_scores, clip_names, outdir):
    """Plot per-frame IoU for each clip and save."""
    fig, axes = plt.subplots(1, len(all_scores), figsize=(6 * len(all_scores), 4),
                             squeeze=False)
    for i, (scores, name) in enumerate(zip(all_scores, clip_names)):
        ax = axes[0, i]
        frames = np.arange(1, len(scores) + 1)
        ax.plot(frames, scores, linewidth=1.0, color="steelblue")
        ax.axhline(y=np.mean([s for s in scores if s > 0]), color="red",
                    linestyle="--", label=f"Mean IoU = {np.mean([s for s in scores if s > 0]):.3f}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("IoU")
        ax.set_title(name)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, "iou_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved IoU curves → {path}")


def plot_summary_bar(all_scores, clip_names, outdir):
    """Bar chart comparing mean IoU across clips."""
    means = []
    track_rates = []
    for scores in all_scores:
        valid = [s for s in scores if s > 0]
        means.append(np.mean(valid) if valid else 0.0)
        track_rates.append(len(valid) / len(scores) * 100 if scores else 0.0)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    x = np.arange(len(clip_names))
    bars = ax1.bar(x - 0.15, means, 0.3, label="Mean IoU", color="steelblue")
    ax1.set_ylabel("Mean IoU")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(clip_names, fontsize=9)

    ax2 = ax1.twinx()
    ax2.bar(x + 0.15, track_rates, 0.3, label="Track Rate %", color="coral", alpha=0.7)
    ax2.set_ylabel("Track Rate (%)")
    ax2.set_ylim(0, 110)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    ax1.set_title("Tracking Performance Across Test Clips")
    ax1.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, "summary_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved summary bar chart → {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate tracker on multiple clips")
    parser.add_argument("--videos", nargs="+", required=True, help="Video paths")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--select-id", type=int, default=0)
    parser.add_argument("--max-lost", type=int, default=30)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    all_scores = []
    clip_names = []
    for vpath in args.videos:
        name = os.path.splitext(os.path.basename(vpath))[0]
        clip_names.append(name)
        out_vid = os.path.join(args.outdir, f"{name}_tracked.mp4")
        print(f"\n{'#'*60}")
        print(f"  Processing: {vpath}")
        print(f"{'#'*60}")
        scores = run_single_object_tracker(
            video_path=vpath,
            output_path=out_vid,
            select_id=args.select_id,
            max_lost_frames=args.max_lost,
            iou_threshold=args.iou_thresh,
            show=False,
        )
        all_scores.append(scores)

    # Generate plots
    plot_iou_curves(all_scores, clip_names, args.outdir)
    plot_summary_bar(all_scores, clip_names, args.outdir)

    # Print table
    print(f"\n{'='*65}")
    print(f"  {'Clip':<20} {'Frames':>8} {'Matched':>8} {'Lost':>8} {'Mean IoU':>10}")
    print(f"  {'-'*60}")
    for name, scores in zip(clip_names, all_scores):
        valid = [s for s in scores if s > 0]
        m = np.mean(valid) if valid else 0.0
        print(f"  {name:<20} {len(scores):>8} {len(valid):>8} "
              f"{len(scores)-len(valid):>8} {m:>10.4f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
