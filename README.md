<div align="center">

# 🛡️ Tri-Sentinel XAI

### Multimodal Explainable AI for Social Engineering Detection

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](https://tri-sentinel-xai.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live Demo → [tri-sentinel-xai.streamlit.app](https://tri-sentinel-xai.streamlit.app/)**

</div>

---

## What is Tri-Sentinel XAI?

Tri-Sentinel XAI is a deployed multimodal AI platform that detects three of the most prevalent social engineering threats in a single interface:

| Module | Threat Detected | Model | XAI Method |
|--------|----------------|-------|------------|
| 🔗 URL Analyzer | Phishing URLs | XGBoost (trained from scratch) | SHAP |
| 💬 NLP Analyzer | Spam / Phishing text | BERT (`cybert79/spamai`) | LIME |
| 🖼️ Image Analyzer | AI-generated deepfakes | ViT (`dima806/deepfake_vs_real_image_detection`) | Grad-CAM |

Every prediction includes a **transparent explanation** — not just a risk score. SHAP shows which URL features triggered detection, LIME highlights which words drove the text classification, and Grad-CAM overlays a heatmap on the image showing which facial regions were suspicious.

---

## Project Structure

```
Tri-Sentinel-XAI/
├── app.py                        # Streamlit UI — three tabs, XAI rendering
├── requirements.txt              # All dependencies
├── modules/
│   ├── url_analyzer.py           # URL feature extraction + XGBoost inference
│   ├── nlp_analyzer.py           # HuggingFace BERT pipeline
│   └── image_analyzer.py         # ViT model inference
├── xai/
│   └── explainer.py              # SHAP, LIME, Grad-CAM
├── Models/
│   ├── url_xgb.ubj               # Trained XGBoost weights (native format)
│   ├── url_preprocessor.pkl      # Fitted sklearn ColumnTransformer
│   └── url_feature_columns.pkl   # Feature column order
└── notebooks/
    └── train_url.ipynb           # XGBoost training notebook (run on Colab)
```

---

## Hardware Requirements

| Spec | Requirement |
|------|-------------|
| **RAM** | 8GB minimum (BERT loads ~1.5GB at inference) |
| **GPU** | Not required — all inference runs on CPU |
| **Disk** | ~1.5GB for HuggingFace model cache (auto-downloaded on first run) |
| **Python** | 3.9, 3.10, or 3.11 (3.12 not fully supported) |
| **OS** | Windows 10+, macOS 12+, Ubuntu 20.04+ |

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/khadijja1/Tri-Sentinel-XAI.git
cd Tri-Sentinel-XAI
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for Windows users:** If `torch` installation fails, install the CPU build explicitly first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first run, the NLP and image models (~440MB + ~350MB) will be downloaded and cached automatically by HuggingFace. This only happens once.

---

## Dataset — URL Module

The URL phishing classifier was trained on the **PhiUSIIL Phishing URL Dataset** (235,795 URLs, 54 features).

**Option 1 — Download from Kaggle:**
```
https://www.kaggle.com/datasets/joebeachcapital/phiusiil-phishing-url
```

**Option 2 — Load without downloading (recommended):**
```python
# No file download needed — streams directly into RAM
pip install ucimlrepo

from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=967)
X = dataset.data.features   # 54 pre-computed features
y = dataset.data.targets    # 1 = Legitimate, 0 = Phishing
```

> Column `FILENAME` can be ignored. `URLSimilarityIndex` was excluded during training due to data leakage risk.

To retrain the model, open `notebooks/train_url.ipynb` and run it in Google Colab (free tier). Save the resulting `url_xgb.ubj`, `url_preprocessor.pkl`, and `url_feature_columns.pkl` to the `Models/` directory.

---

## Tech Stack

- **XGBoost** — URL phishing classification (custom trained)
- **HuggingFace Transformers + PyTorch** — NLP and image inference
- **scikit-learn** — Preprocessing pipeline
- **SHAP** — URL module explainability
- **LIME** — NLP module explainability
- **Grad-CAM** — Image module explainability
- **Streamlit** — Web UI and cloud deployment
- **Pillow / OpenCV** — Image preprocessing and heatmap rendering

---

## Models Used

| Model | Source | Task |
|-------|--------|------|
| `cybert79/spamai` | [HuggingFace](https://huggingface.co/cybert79/spamai) | Spam/phishing text classification |
| `dima806/deepfake_vs_real_image_detection` | [HuggingFace](https://huggingface.co/dima806/deepfake_vs_real_image_detection) | Real vs AI-generated face detection |

---

## Limitations

- The URL module imputes webpage-source features (e.g., `LineOfCode`, `HasTitle`) with median values at inference time since fetching the live page is not performed during classification.
- The NLP module supports **English only**. Urdu and other languages are outside the training distribution.
- The deepfake model was trained on data from approximately 2021–2022. Images generated by newer models may show reduced detection accuracy.

---

## References

1. UCI ML Repository — PhiUSIIL Phishing URL Dataset (ID 967)
2. HuggingFace — `cybert79/spamai`
3. HuggingFace — `dima806/deepfake_vs_real_image_detection`
4. Ogunlade, O. T. (2025). Role of AI in Detection of Social Engineering Attacks. *European Journal of Technology*, 9(1).
5. Soundararajan & Xu (2026). Deepfake Detection Using SigLIP-2 Vision Transformers. *AI*, 7(3).

---

<div align="center">

OWNER: Khadija Faisal 
khadijafaysal444@gmail.com

</div>
