# SWTVIT — Spatio-Frequency Deepfake Detection

[![Frame Accuracy](https://img.shields.io/badge/Frame%20Accuracy-84.61%25-blue)]()
[![Video Accuracy](https://img.shields.io/badge/Video%20Accuracy-90.00%25-brightgreen)]()
[![Macro F1](https://img.shields.io/badge/Macro%20F1-85.09%25-blue)]()
[![Dataset](https://img.shields.io/badge/Dataset-FaceForensics%2B%2B-orange)]()
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red)]()
[![Open In Colab](https://colab.research.google.com/github/Annanya-pS/Video_Forgery_Detection_SWTVIT/blob/main/Complete_deepfake_detection_pipeline_with_UI.ipynb)

> **4-class deepfake detection** using Stationary Wavelet Transform + Vision Transformer (ViT-Small) with GradCAM spatial localization. Trained and evaluated on FaceForensics++ (c23).

**Authors:** Annanya Priyadarshini Sahoo . Shiv Vendra Singh 
**Institution:** Jaypee Institute of Information Technology, Noida  
**Supervisor:** Prof. Vineet Khandelwal

---

## Overview

SWTVIT is a deepfake detection pipeline that goes beyond binary real/fake classification. It distinguishes between **four manipulation types** and visually highlights *where* the manipulation occurred in each frame.

**Pipeline:** Video → Frame Extraction (ffmpeg) → Face Detection (MTCNN) → Stationary Wavelet Transform → ViT-Small → GradCAM Localization

| Stage | Tool / Method |
|---|---|
| Frame extraction | ffmpeg @ 1 fps, max 50 frames/video |
| Face detection & alignment | MTCNN (facenet-pytorch) |
| Frequency preprocessing | Stationary Wavelet Transform (Haar, level 1) |
| Classification backbone | ViT-Small patch16/224, pretrained ImageNet-21k |
| Spatial localization | GradCAM on final LayerNorm |
| Video-level decision | Soft-voting (averaged softmax probabilities) |

---

## Results

### Comparison with prior work on FF++ (c23)

| Method | Backbone | Classes | Accuracy |
|---|---|---|---|
| MesoNet | Lightweight CNN | 2 (binary) | 83.10%† |
| XceptionNet | Xception | 2 (binary) | 89.30%† |
| F3Net | ResNet-50 | 2 (binary) | 90.43%† |
| SPSL | ResNet | 2 (binary) | 91.50%† |
| ViT-B | ViT-Base | 2 (binary) | 90.64%† |
| **SWTVIT (ours)** | **ViT-Small** | **4 classes** | **90.00%** |

† frame-level accuracy on binary classification. SWTVIT reports video-level accuracy on 4-class problem — a strictly harder task.

### SWTVIT consolidated metrics

| Metric | Frame level | Video level |
|---|---|---|
| Accuracy | 84.61% | **90.00%** |
| Macro Precision | 84.88% | 89.96% |
| Macro Recall | 85.52% | 90.00% |
| Macro F1 | 85.09% | 89.89% |
| Real — F1 | 0.784 | 0.842 |
| Deepfake — F1 | 0.864 | 0.900 |
| FaceSwap — F1 | 0.904 | 0.927 |
| Face2Face — F1 | 0.852 | 0.927 |

Best validation accuracy: **84.81%** at epoch 16 (out of 25 epochs).

### Real-world generalization (out-of-distribution personal videos)

| Video | True class | Predicted | Frames | Frame consensus |
|---|---|---|---|---|
| my_real_video.mp4 | REAL | **REAL** ✅ | 11 | Predominantly REAL |
| my_faceswap_video.mp4 | FACESWAP | **FACESWAP** ✅ | 10 | Unanimous FACESWAP |

The face-swap video was produced with a **commercial tool not present in the training set**, confirming that SWT captures generalizable frequency-domain artifacts rather than dataset-specific patterns.

---

## GradCAM Localization

The model doesn't just classify — it highlights *where* manipulation occurred.

- **Deepfake frames:** attention concentrates along the facial boundary and nasal bridge (autoencoder blending seam)
- **FaceSwap frames:** attention concentrates in the central facial region (3D morphable model texture boundary)

See `assets/` for example GradCAM outputs from the report.

---

## Setup & Usage

### 1. Open in Colab (recommended)

Click the **Open in Colab** badge above. The notebook is fully self-contained and runs on a free T4 GPU.

### 2. Dataset

The FaceForensics++ dataset requires registration. Request access at:  
https://github.com/ondyari/FaceForensics

After downloading, update `FF_SOURCES` in Cell 3 of the notebook to point to your data paths. The notebook expects this structure:

```
data/
├── original_sequences/youtube/c23/videos/       ← Real
├── manipulated_sequences/Deepfakes/c23/videos/  ← Deepfakes
├── manipulated_sequences/FaceSwap/c23/videos/   ← FaceSwap
└── manipulated_sequences/Face2Face/c23/videos/  ← Face2Face
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Model weights

The trained `best_model.pth` checkpoint (epoch 16, val_acc=84.81%) is too large for GitHub.  
Download it from Google Drive: **[best_model.pth — ~85 MB](YOUR_DRIVE_LINK_HERE)**

Place it at: `checkpoints/best_model.pth`

### 5. Run inference on a single video

```python
# Inside the notebook — Cell 17
result = infer_single_video(
    video_path   = 'path/to/your/video.mp4',
    model        = MODEL,
    detector     = MTCNN_DETECTOR,
    transform    = EVAL_TRANSFORMS,
    device       = DEVICE,
)
print(result['prediction'])  # e.g. 'faceswap'
```

### 6. Run the Gradio UI

The last cell in the notebook launches an interactive Gradio interface where you can upload any video and get a real-time prediction with confidence scores.

```
Verdict        : 🚨 MANIPULATED
Prediction     : FACESWAP
Confidence     : 94.2%
──────────────────────────────────────────────────
Frames analysed: 10 / 10
Frame votes    :
  FACESWAP       10x  ████████████████████
```

---

## Notebook Structure

| Cell | Description |
|---|---|
| 1 | Mount Google Drive, install dependencies |
| 2 | Imports, global config (`CFG` dict), reproducibility seed |
| 3 | Dataset discovery, video-level stratified split (70/15/15) |
| 4 | Frame extraction via ffmpeg |
| 5 | Face detection & alignment (MTCNN) |
| 6 | SWT preprocessing (Haar, level 1) |
| 7 | Dataset class & DataLoader |
| 8 | SWTVIT model definition (ViT backbone + classification head) |
| 9 | Training setup (AdamW, warmup + cosine annealing, label smoothing) |
| 10 | Training loop with early stopping & checkpoint saving |
| 11–12 | Dataset statistics & sample visualization |
| 13 | Training curves |
| 14 | Frame-level evaluation |
| 15 | Video-level evaluation (soft-voting) |
| 16 | Confusion matrix visualization |
| 17 | Single video inference demo |
| 18 | Export & final pipeline summary |
| 19+ | GradCAM localization, corresponding video set, personal video test |
| Last | Gradio web UI |

---

## Architecture

```
Input video
    │
    ▼ ffmpeg @ 1 fps
Frames (PNG)
    │
    ▼ MTCNN face detection + alignment
Face crops (224×224)
    │
    ▼ Stationary Wavelet Transform (Haar, level 1)
    │  G → {cA, cH, cV, cD}  →  C = √(cH²+cV²+cD²)  →  min-max normalize  →  3-channel stack
    │
SWT feature tensor (224×224×3)
    │
    ▼ ViT-Small patch16/224 (ImageNet-21k pretrained, 21.7M params)
    │  196 patches + [CLS] token → 12 Transformer blocks → [CLS] embedding (dim=384)
    │
    ▼ Classification head (LayerNorm → Dropout(0.3) → Linear(384→4))
    │
Per-frame logits → softmax probabilities
    │
    ▼ Soft-voting across frames
Video-level prediction: {Real, Deepfake, FaceSwap, Face2Face}
    │
    ▼ GradCAM (last LayerNorm of final Transformer block)
Spatial heatmap + bounding box overlay
```

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Backbone LR | 1e-5 (10× lower than head) |
| Head LR | 1e-4 |
| Weight decay | 1e-2 |
| LR schedule | Linear warmup (5 ep) → cosine annealing |
| Loss | Cross-entropy + label smoothing (ε=0.05) |
| Batch size | 32 |
| Max epochs | 25 |
| Early stopping | Patience = 8 |
| Best checkpoint | Epoch 16 |
| Gradient clipping | max norm = 1.0 |
| Data augmentation | H-flip (p=0.5), 90° rot (p=0.5), Gaussian blur (p=0.2), color jitter (p=0.3) |

---

## Repository Structure

```
SWTVIT/
├── SWTVIT_pipeline.ipynb     ← Complete pipeline (train + evaluate + Gradio UI)
├── README.md
├── requirements.txt
├── .gitignore
├── CITATION.cff
├── assets/
│   ├── gradcam_deepfake.png  ← GradCAM output: Deepfake frame
│   ├── gradcam_faceswap.png  ← GradCAM output: FaceSwap frame
│   ├── training_curves.png
│   └── confusion_matrix.png
└── docs/
    ├── project_report.pdf
    └── dataset_setup.md
```

---

## Citation

```bibtex
@misc{singh2026swtvit,
  title        = {Spatio-Frequency Analysis of Facial Regions for Video Forgery Detection},
  author       = {Singh, Shiv Vendra and Sahoo, Annanya Priyadarshini},
  year         = {2026},
  institution  = {Jaypee Institute of Information Technology, Noida},
  note         = {Minor Project, B.Tech ECE, 6th Semester}
}
```

---

## License

This project is released under the MIT License — see `LICENSE` for details.
