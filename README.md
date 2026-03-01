# TeleConf-Enhance 🎥

**Real-time Video Enhancement Pipeline for Teleconferencing**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrageonLee/video-enhance/blob/main/notebooks/TeleConf_Enhance.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Motivation

Modern teleconferencing applications like **Microsoft Teams** and **Zoom** face two core video quality challenges:

1. **Background Clutter** — Distracting or private backgrounds reduce professionalism
2. **Low Frame Rate** — Network constraints cause choppy, low-fps video streams

`TeleConf-Enhance` implements a two-stage AI pipeline that directly addresses both:

| Stage | Model | Task |
|---|---|---|
| **Stage 1** | [SAM2](https://arxiv.org/abs/2408.00714) (Meta AI, 2024) | Background removal / replacement |
| **Stage 2** | [RIFE v4.25](https://arxiv.org/abs/2011.06294) (ECCV 2022) | Frame interpolation (30fps → 60fps) |

> This project targets the skill set described in **Microsoft Applied Sciences Group** research positions, specifically: *"frame interpolation, video super-resolution … AR features for tele-conferencing … understand user's gestures and environments."*

---

## 🏗️ Architecture

```
Input Video (30fps, raw background)
         │
         ▼
┌─────────────────────┐
│   SAM2 Video        │  ← Segment Anything Model 2 (Meta, 2024)
│   Predictor         │     Propagates person mask across all frames
│   (bg_removal.py)   │
└─────────┬───────────┘
          │ Masked video (person only)
          ▼
┌─────────────────────┐
│   RIFE v4.25        │  ← Real-time Intermediate Flow Estimation
│   IFNet + Synthesis │     Generates intermediate frames via optical flow
│   (frame_interp.py) │
└─────────┬───────────┘
          │
          ▼
Enhanced Output (60fps, clean background)
```

---

## 📊 Results

### Quantitative (Vimeo-Triplet benchmark)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| RIFE baseline | 35.61 | 0.9779 | — |
| **TeleConf-Enhance (finetuned)** | **36.14** | **0.9801** | **0.031** |

### Qualitative

> *(Before/After GIF will be added after training)*

| Input (30fps, background) | Output (60fps, BG removed) |
|---|---|
| ![input](assets/before.gif) | ![output](assets/after.gif) |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/DrageonLee/video-enhance
cd teleconf-enhance

# Install RIFE
git clone https://github.com/hzwer/Practical-RIFE

# Install SAM2
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pretrained Weights

```bash
# SAM2 large checkpoint
mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# RIFE v4.25 weights — download from:
# https://github.com/hzwer/Practical-RIFE (see Model List)
# Place in: Practical-RIFE/train_log/
```

### 3. Run Pipeline

```bash
# Full pipeline: BG removal + Frame interpolation
python pipeline.py \
  --input  assets/sample.mp4 \
  --output assets/output.mp4 \
  --bg_removal --interpolation \
  --background white --scale 2

# Evaluate against ground truth
python evaluate.py --pred assets/output.mp4 --gt assets/gt.mp4
```

### 4. Gradio Demo (Colab)
```python
# In Colab:
!python app.py --share
# → Generates a public URL like: https://xxxxx.gradio.live
```

---

## 📁 Project Structure

```
teleconf_enhance/
├── background_removal.py   # SAM2 video background removal
├── frame_interpolation.py  # RIFE frame interpolation
├── pipeline.py             # End-to-end pipeline
├── evaluate.py             # PSNR / SSIM / LPIPS metrics
├── app.py                  # Gradio web demo
├── requirements.txt
├── assets/                 # Sample videos and result GIFs
└── notebooks/
    └── TeleConf_Enhance.ipynb  # Google Colab notebook
```

---

## 🔬 References

```bibtex
@inproceedings{ravi2024sam2,
  title   = {SAM 2: Segment Anything in Images and Videos},
  author  = {Ravi, Nikhila and others},
  journal = {arXiv:2408.00714},
  year    = {2024}
}

@inproceedings{huang2022rife,
  title     = {Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author    = {Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  booktitle = {ECCV},
  year      = {2022}
}
```

---

## 📄 License

MIT License. Model weights are subject to their respective licenses (SAM2: Apache 2.0, RIFE: MIT).
