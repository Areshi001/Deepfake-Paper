# Regional Sensitivity and Multi-Signal Forensic Evidence in Deepfake Detection: An Empirical Study and Conceptual Framework

This repository contains the Google Colab notebook and supporting code used for the **executed empirical experiments (Component A)** of the paper above. It is published for **reproducibility of the experimental work** only.

## Overview

The study compares full-face and localized facial-region inputs for deepfake detection on FaceForensics++ (C23) using lightweight architectures:

- **Full-face baseline** — MobileNetV3-Small
- **Periocular region** — EfficientNet-B0
- **Mouth region** — EfficientNet-B0
- **Cheek regions** — EfficientNet-B0
- **Temporal extension** — MobileNetV3-Small feature extractor + single-layer GRU

The periocular model achieved the **highest observed AUC of 0.8681**, compared with **0.7872** for the full-face baseline (see [Results](#results)).

> **Important caveats:** All reported results are from **single training runs**. No statistical significance testing or multi-seed validation was performed. The paper additionally proposes a theoretical multi-signal framework (Component B); that framework is **not implemented or experimentally validated** in this repository (see [Component B](#component-b)).

## Paper

**Title:** Regional Sensitivity and Multi-Signal Forensic Evidence in Deepfake Detection: An Empirical Study and Conceptual Framework

**Author:** Abu Henaf Rashid Ahmad Mufti — Department of Computer Science & Engineering, United International University (UIU), Dhaka, Bangladesh

**Paper URL / DOI:** *To be added when available.*

An earlier version of the Component A empirical work appears in the proposal *"Lightweight Deepfake Detection Through Facial Region Sensitivity Analysis Under H.264 C23 Compression"* by the same author.

## Dataset

**Dataset:** FaceForensics++ C23 — Kaggle mirror
**URL:** <https://www.kaggle.com/datasets/xdxd003/ff-c23>

- The experiments use only the **`Deepfakes/`** and **`original/`** folders under **C23 (H.264) compression**.
- FaceForensics++ was created by Rössler et al. (ICCV 2019). The author of this repository **does not own, host, or claim any rights over** FaceForensics++ or the Kaggle mirror. Users are responsible for complying with the original dataset's terms and the mirror's terms when downloading or using the data.

## Experimental Setup

| Setting | Value |
| --- | --- |
| Input image size | 224 × 224 |
| Full-face backbone | MobileNetV3-Small (ImageNet-pretrained) |
| Regional backbone | EfficientNet-B0 (ImageNet-pretrained) |
| Face landmarks | dlib 68-point predictor |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Scheduler | OneCycleLR |
| Mixed precision | FP16 / AMP |
| Batch size | 32 |
| Early stopping | Patience of 3 epochs (selection on validation AUC) |
| Temporal model | Single-layer GRU |
| GRU hidden size | 128 |
| Sequence length | 8 frames |
| Frame stride | 4 initially, with adaptive stride evaluation |

Regional crops are derived from the 68-point landmarks: periocular (points 36–47), mouth (points 48–67), and cheeks (points 1–5 and 11–15), each expanded by 15 pixels and resized to 224 × 224.

> **Note:** The YAML files bundled under `configs/` ship with slightly different defaults (`lr: 3e-4`; spatial models use `batch_size: 64`). The values listed above are the ones reported in the paper.

## Compute

The experiments were executed on:

- **Google Colab** with an **NVIDIA T4** GPU
- Peak system RAM: **≈ 6.5 GB**
- Peak GPU VRAM: **≈ 0.9 GB**
- FP16 mixed precision

Training wall-clock time was not recorded and is therefore not reported.

## Results

Test results on FaceForensics++ C23, as reported in the paper:

| Model | Accuracy | Precision | Recall | AUC |
| --- | ---: | ---: | ---: | ---: |
| Full Face (MobileNetV3-Small) | 0.6925 | 0.6831 | 0.7012 | 0.7872 |
| Mouth (EfficientNet-B0) | 0.7573 | 0.7402 | 0.7504 | 0.8453 |
| Cheeks (EfficientNet-B0) | 0.7627 | 0.7453 | 0.7572 | 0.8558 |
| GRU Temporal | 0.7600 | 0.7460 | 0.7870 | 0.8654 |
| Periocular / Eyes (EfficientNet-B0) | 0.7731 | 0.7482 | 0.7846 | **0.8681** |

The **periocular** classifier produced the **highest observed AUC (0.8681)**, against 0.7872 for the full-face baseline.

These are single-run experimental results; no confidence intervals or significance tests were computed.

## Repository Structure

```text
Deepfake-Paper/
├── README.md
├── requirements.txt
├── .gitignore
├── Deepfake_Detection_FINAL.ipynb    # Colab notebook — full executed pipeline
│
├── src/
│   ├── __init__.py
│   ├── dataset.py                    # Kaggle data handling, frame extraction, video-level split
│   ├── preprocessing.py              # Train/validation transforms
│   ├── extract_regions.py            # dlib-based periocular / mouth / cheek crop extraction
│   ├── models.py                     # MobileNetV3-Small, EfficientNet-B0, GRU model builders
│   ├── train.py                      # Training loop (AMP, OneCycleLR, early stopping)
│   ├── evaluate.py                   # Checkpoint evaluation (accuracy / precision / recall / AUC)
│   └── utils.py                      # Shared helpers, device and directory constants
│
├── configs/
│   ├── baseline.yaml                 # Full-face MobileNetV3-Small
│   ├── periocular.yaml               # Periocular EfficientNet-B0
│   ├── mouth.yaml                    # Mouth EfficientNet-B0
│   ├── cheeks.yaml                   # Cheek EfficientNet-B0
│   └── gru.yaml                      # Temporal GRU
│
└── outputs/
    ├── figures/
    ├── results/
    └── sample_predictions/
```

`Deepfake_Detection_FINAL.ipynb` is the primary executed artifact; `src/` and `configs/` provide the same pipeline in modular form.

## How to Run

### Option A — Google Colab (primary route; this is how the experiments were executed)

1. Open `Deepfake_Detection_FINAL.ipynb` in Google Colab.
2. Set the runtime to GPU: **Runtime → Change runtime type → T4 GPU**.
3. Obtain the FaceForensics++ C23 Kaggle mirror (`https://www.kaggle.com/datasets/xdxd003/ff-c23`) and make the `Deepfakes/` and `original/` folders available to the notebook (e.g., Google Drive mount or Kaggle API with your own `kaggle.json` token). Update the dataset path in the notebook to your `<DATASET_ROOT>` location.
4. Download the dlib 68-point landmark model required by the regional pipeline:

   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   ```

5. Run the notebook cells in order: frame extraction → dataset split → regional crop extraction → training → evaluation.
6. Artifacts are written under `outputs/` (figures, results, sample predictions).

### Option B — Local / modular scripts

```bash
git clone https://github.com/Areshi001/Deepfake-Paper
cd Deepfake-Paper

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

- Dependencies (unpinned) are listed in `requirements.txt` (PyTorch, torchvision, OpenCV, NumPy, pandas, scikit-learn, Pillow, dlib, PyYAML, matplotlib, tqdm, ffmpeg-python, timm).
- FFmpeg/ffprobe must be available on the PATH (used for frame extraction).
- Download the dlib landmark model as shown above.

Train and evaluate each configuration:

```bash
python src/train.py --config configs/baseline.yaml      # Full-face baseline
python src/train.py --config configs/periocular.yaml    # Periocular
python src/train.py --config configs/mouth.yaml         # Mouth
python src/train.py --config configs/cheeks.yaml        # Cheeks
python src/train.py --config configs/gru.yaml           # Temporal GRU

python src/evaluate.py --checkpoint outputs/models/<model_alias>_best.pth --config configs/<config>.yaml
```

## Reproducibility Notes

- The dataset split is performed **at the video level** (stratified 75/15/10 train/val/test), so no video contributes frames to more than one split.
- **20 evenly spaced frames** are sampled per video.
- **dlib landmark failures** can reduce the effective number of regional crops available for training and evaluation.
- The regional and full-face comparisons therefore **do not use perfectly identical effective training data**; comparisons should be read with this in mind.
- All reported results are based on **single training runs** — re-running the pipeline will produce approximately, but not exactly, the reported numbers.

## Limitations

The limitations stated in the paper apply to all results in this repository:

1. Only the **DeepFakes** manipulation type was evaluated.
2. Only **C23 compression** was evaluated.
3. **No multi-seed confidence intervals** or formal significance tests were performed.
4. **dlib landmark failures** reduce regional sample counts.
5. The temporal GRU adaptive-stride loop **did not reach the target threshold**.
6. **Component B is conceptual** and has not been experimentally implemented.

## Component B

The paper also proposes a theoretical multi-signal forensic framework (Component B) covering spatial, frequency/residual, compression/acquisition, semantic/identity, and statistical/auxiliary signals.

> "This repository currently reproduces the executed empirical Component A experiments. The proposed Component B framework is conceptual and is not implemented or experimentally validated in this repository."

## Citation

If you use this work, please cite the paper:

```bibtex
@misc{mufti_regional,
  author = {Abu Henaf Rashid Ahmad Mufti},
  title  = {Regional Sensitivity and Multi-Signal Forensic Evidence in Deepfake Detection:
            An Empirical Study and Conceptual Framework},
  year   = {2026},
  note   = {Paper URL/DOI to be added upon publication}
}
```

The earlier Component A proposal can be cited as:

```bibtex
@misc{mufti_componentA,
  author = {Abu Henaf Rashid Ahmad Mufti},
  title  = {Lightweight Deepfake Detection Through Facial Region Sensitivity Analysis
            Under {H.264} {C23} Compression},
  year   = {2026},
  note   = {Component A research proposal, United International University (UIU), Dhaka, Bangladesh}
}
```

## License

License: To be determined.
