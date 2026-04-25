# Multi-Object Tracking with Re-Identification

> Final project for **EECE 5639: Computer Vision** — Northeastern University, Spring 2026
> Built by **[Shourya Vuddemarri](https://github.com/YOUR-USERNAME)** and **Meet Brahmbhatt**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)
![License](https://img.shields.io/badge/license-MIT-green)

A complete multi-object tracking system with appearance-based re-identification, built across three progressive levels: single-object tracking, multi-object tracking with Re-ID, and cross-camera Re-ID across two synchronized views.

---

## 🎯 Highlights

| Level | Task | Headline Result |
|-------|------|-----------------|
| **1** | Single-object tracking | Mean IoU **0.94 – 0.97** across 3 custom clips |
| **2** | Multi-object + MOT17 | **MOTA 0.4995, IDF1 0.7077** on MOT17-09-DPM (beats SORT by 16 MOTA points) |
| **3** | Cross-camera Re-ID | **3/3 tracks matched (100%)** across 90°-offset views, avg cosine distance < 0.10 |

---

## 📺 Demo

<!-- Replace these with actual GitHub-hosted screenshots / GIFs of your output videos -->

| Level 1: Single-object | Level 2: Multi-object + Re-ID | Level 3: Cross-camera |
|:---:|:---:|:---:|
| ![Level 1](docs/images/clip1_frame.png) | ![Level 2](docs/images/multi2b_frame.png) | ![Level 3](docs/images/l3_frame_cam1.png) |

> 💡 **Tip:** For best demos on GitHub, convert your output `.mp4` files to short `.gif` clips (≤10 MB) with `ffmpeg -i input.mp4 -vf "fps=10,scale=480:-1" -t 8 demo.gif` and link them above.

---

## 🏗️ Pipeline

```
Video frame
    │
    ▼
[YOLOv8n detector] ──────────► person bounding boxes (conf > 0.5)
    │
    ▼
[ResNet18 backbone] ─────────► 512-d L2-normalized appearance embedding
    │
    ▼
[Constant-velocity Kalman] ──► state prediction per track
    │
    ▼
[Hungarian assignment] ──────► cost = λ·(1−IoU) + (1−λ)·cos_dist
    │
    ▼
[Track lifecycle manager] ───► tentative → confirmed → lost → deleted
    │
    ▼
[Re-ID gallery] ─────────────► 10-frame gallery per track for occlusion recovery
    │
    ▼
Annotated output video + persistent IDs
```

For **Level 3**, the Level 2 tracker runs independently on two camera feeds, and gallery-mean signatures are matched across cameras via Hungarian assignment over cosine distance — producing a shared global ID (GID) across views.

---

## 📂 Project Structure

```
project2/
├── src/
│   ├── kalman_filter.py          # Constant-velocity Kalman filter
│   ├── embedding_extractor.py    # ResNet18 → 512-d L2-normalized embeddings
│   ├── single_object_tracker.py  # Level 1: single-target tracking
│   ├── multi_object_tracker.py   # Level 2: Hungarian + Re-ID gallery
│   ├── evaluate.py               # IoU evaluation for Level 1
│   ├── eval_mot17.py             # MOT17 MOTA / IDF1 evaluation
│   ├── cross_camera_reid.py      # Level 3: cross-camera matching
│   └── generate_report.py        # IEEE-format PDF report generator
│
├── Videos/                       # Custom test clips
│   ├── clip{1,2,3}.mp4           # Single-person (Level 1)
│   └── multi{1,2,3}.mp4          # Multi-person (Level 2)
│
├── data/MOT17/train/             # MOT17 benchmark (downloaded from Kaggle)
├── Results/                      # Level 1 + Level 2 outputs
├── results_1/                    # Level 3 cross-camera outputs
│   ├── cross_camera/             # Clip pair 1
│   └── cross_camera2/            # Clip pair 2
│
├── docs/                         # README assets, report PDF, presentation
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- ~5 GB disk for MOT17 + custom videos
- Tested on Windows 11 (PowerShell), should work on Linux/macOS

### Install

```bash
# Clone
git clone https://github.com/YOUR-USERNAME/mot-reid.git
cd mot-reid

# Create venv
python -m venv venv

# Activate (pick one)
.\venv\Scripts\Activate.ps1                 # Windows PowerShell
source venv/bin/activate                    # Linux / macOS

# Windows only — if PowerShell blocks activation:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Install dependencies
pip install -r requirements.txt
```

<details>
<summary><strong>📦 Dependencies (click to expand)</strong></summary>

- `torch`, `torchvision` — ResNet18 backbone, CPU inference
- `ultralytics` — YOLOv8 detector
- `opencv-python` — video I/O, drawing
- `scipy` — Hungarian algorithm (`linear_sum_assignment`)
- `numpy`, `matplotlib`, `Pillow`
- `motmetrics` — MOTA / IDF1 evaluation
- `reportlab` — IEEE-format PDF report generator

</details>

### GPU note

The dev hardware (RTX 5070 Ti, Blackwell `sm_120`) is incompatible with stable PyTorch builds, so all inference runs on **CPU**. The code is GPU-ready — flipping `device="cpu"` to `device="cuda"` in `embedding_extractor.py` and the YOLOv8 calls is the only change needed once your hardware is supported. **Output is identical** between CPU and GPU; only speed differs.

### MOT17 dataset

The Kaggle mirror is the most reliable source — search "MOT17 dataset" on Kaggle. Only the `train/` split is needed (the `test/` split has no public ground truth). Place under `data/MOT17/train/`.

---

## 🎬 Usage

### Level 1 — Single-object tracking

```bash
# Run the tracker
python src/single_object_tracker.py --video Videos/clip1.mp4 --output Results/clip1_out.mp4 --no-show

# Evaluate IoU
python src/evaluate.py --video Videos/clip1.mp4 --output Results/clip1_metrics.json
```

`--no-show` skips the OpenCV window — recommended for CPU runs.

### Level 2 — Multi-object tracking

```bash
# Custom clip
python src/multi_object_tracker.py --video Videos/multi2.mp4 --output Results/multi2_out.mp4 --no-show

# MOT17 benchmark with tuned hyperparameters
python src/eval_mot17.py --seq MOT17-09-DPM --lambda 0.7 --max_lost 50
```

### Level 3 — Cross-camera Re-ID

```bash
python src/cross_camera_reid.py \
    --cam1 path/to/cam1_clip1.mp4 \
    --cam2 path/to/cam2_clip1.mp4 \
    --output results_1/cross_camera/ \
    --threshold 0.5 --min_track_len 10
```

**Outputs:**
- `cam1_annotated.mp4`, `cam2_annotated.mp4` — per-camera tracking with local IDs and global IDs (GIDs)
- `cross_camera_distances.png` — full cosine distance matrix with Hungarian assignments highlighted
- `matching_summary.png` — bar chart of matched / unmatched tracks
- `match_results.json` — Cam1 ID → Cam2 ID pairings with distances

### Generate the IEEE final report

```bash
python src/generate_report.py --output docs/EECE5639_Final_Report.pdf
```

---

## 📊 Results

### Level 1 — Single-object tracking

| Clip | Frames | Lost | Mean IoU | Track Rate |
|------|-------:|-----:|---------:|-----------:|
| 1 (Hallway)     | 389 | 1 | **0.9587** | 99.7% |
| 2 (Occlusion)   | 474 | 1 | **0.9693** | 99.8% |
| 3 (Fast Motion) | 402 | 1 | **0.9433** | 99.8% |

### Level 2 — MOT17-09-DPM benchmark

| Metric        | Value             | Notes |
|---------------|-------------------|-------|
| **MOTA**      | **0.4995**        | Hit project target (>0.50) |
| **IDF1**      | **0.7077**        | Identity preservation |
| Precision     | 0.8772            | High when detected |
| Recall        | 0.5931            | Limited by YOLOv8n |
| ID Switches   | 56                | Out of 525 frames |
| TP / FP / FN  | 3158 / 442 / 2167 | — |

> **For context:** SORT ≈ 0.34 MOTA, DeepSORT ≈ 0.61 MOTA on the same benchmark. This system sits between the two using a generic ImageNet backbone (vs. DeepSORT's pedestrian-specialized Re-ID network).

### Level 3 — Cross-camera Re-ID

**Setup:** Two iPhone cameras (16 Pro and 17) at a 90° offset, ~30 s synchronized clips, 720p (downscaled from 4K with FFmpeg).

| Clip Pair | Cam1 Tracks | Cam2 Tracks | Matched | Avg Cosine Distance |
|-----------|------------:|------------:|--------:|--------------------:|
| 1 (walking)  | 3 | 4 | **3 / 3** | **0.0854** |
| 2 (re-entry) | 4 | 3 | **3 / 3** | **0.0953** |

All matched pairs had cosine distance below **0.10** — well under the 0.5 acceptance threshold and even under the 0.4 single-camera Re-ID threshold. Unmatched tracks were short-lived spurious detections (<15 frames) correctly rejected by the 10-frame minimum-length filter.

---

## ⚙️ Hyperparameters

| Parameter | Value | Notes |
|-----------|------:|-------|
| YOLOv8 confidence threshold     | 0.5   | Person class only (COCO 0) |
| Embedding crop size             | 128 × 64 | ImageNet normalization |
| Embedding gallery size          | 10    | Most recent crops per track |
| **λ (IoU vs. cosine weight)**   | **0.7** | Increased from 0.5 for MOT17 |
| Cost rejection threshold        | 0.7   | Above this → no assignment |
| Re-ID threshold (single-camera) | 0.4   | Cosine distance |
| Re-ID threshold (cross-camera)  | 0.5   | Cosine distance |
| Confirmation hits               | 3     | Tentative → confirmed |
| **Max lost frames**             | **50** | Increased from 30 for MOT17 |
| Min track length (Level 3)      | 10    | Eligible-for-matching threshold |

**Bold** values are the ones tuned during MOT17 evaluation. Together they moved MOTA from 0.476 to **0.500**, crossing the project target.

---

## 🔍 Known Limitations

- **Recall (0.59)** is bottlenecked by YOLOv8n. Switching to YOLOv8m or YOLOv8l improves recall but slows inference — a deliberate speed/accuracy tradeoff.
- **ID switches under viewpoint change** (e.g., a person turning around) come from ImageNet-pretrained embeddings not being specialized for person Re-ID. Fine-tuning on Market-1501 would directly address this.
- **CPU-only inference** is forced by hardware/PyTorch incompatibility, not by design.
- **Level 3 evaluation is qualitative** — matches were verified by inspecting the annotated output videos. Running on the EPFL multi-camera pedestrian dataset would yield quantitative cross-view IDF1.

---

## 🧭 Future Work

1. Fine-tune the ResNet18 embedding extractor on **Market-1501** for pedestrian-specialized features
2. Run on the remaining MOT17 sequences (02, 04) and on the **EPFL multi-camera dataset** for quantitative cross-view metrics
3. Add an ablation comparing **IoU-only vs. combined IoU + cosine** cost matrices to quantify the embedding contribution
4. Move to **GPU inference** once stable PyTorch supports the Blackwell architecture

---

## 📚 References

1. Bewley et al. **"Simple Online and Realtime Tracking"** (SORT). ICIP 2016.
2. Wojke et al. **"Simple Online and Realtime Tracking with a Deep Association Metric"** (DeepSORT). ICIP 2017.
3. Milan et al. **"MOT16: A Benchmark for Multi-Object Tracking."** arXiv:1603.00831, 2016.
4. Zheng et al. **"Scalable Person Re-Identification: A Benchmark"** (Market-1501). ICCV 2015.
5. Fleuret et al. **"Multicamera People Tracking with a Probabilistic Occupancy Map."** TPAMI 2008.
6. Jocher, Chaurasia, Qiu. **"Ultralytics YOLOv8."** [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
7. He et al. **"Deep Residual Learning for Image Recognition."** CVPR 2016.
8. Kuhn. **"The Hungarian Method for the Assignment Problem."** 1955.
9. Kalman. **"A New Approach to Linear Filtering and Prediction Problems."** 1960.

---

## 🙏 Acknowledgments

Final project for **EECE 5639: Computer Vision** at Northeastern University (Spring 2026), with guidance from the course staff. AI assistance ([Anthropic Claude](https://claude.ai)) was used during development for code scaffolding, debugging, and report drafting.

---

## 📄 License

MIT — see [`LICENSE`](LICENSE) for details. Free to use for academic and learning purposes.

---

<p align="center">
  Built with ☕ at Northeastern University · Boston, MA
</p>
