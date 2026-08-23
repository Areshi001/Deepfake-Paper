# Lightweight Deepfake Detection Through Facial Region Sensitivity Analysis Under H.264 C23 Compression

## Research problem

This project investigates whether localized facial regions can provide stronger deepfake detection signals than full-face inputs under compressed video conditions, while maintaining low computational requirements.

The study focuses on FaceForensics++ C23 and compares full-face, periocular, mouth, and cheek inputs using lightweight convolutional neural networks.

A lightweight temporal GRU model is also evaluated as a secondary comparison against spatially localized detection.

## Research question

Which facial region carries the strongest deepfake detection signal when different facial regions are evaluated under the same compressed-video condition using lightweight models?

A secondary question is whether lightweight temporal aggregation can outperform the strongest static facial-region classifier.

## Proposed approach

The implementation evaluates five model configurations:

1. MobileNetV3 Small on full-face inputs.
2. EfficientNet-B0 on periocular regions.
3. EfficientNet-B0 on mouth regions.
4. EfficientNet-B0 on cheek regions.
5. MobileNetV3 Small feature extractor followed by a GRU for temporal modeling.

All region-specific models use the same general training protocol so that the comparison focuses on the discriminative value of the facial regions.

## Dataset

The experiments use FaceForensics++ C23.

The implementation focuses on:

- `deepfakes/`
- `original/`

The study uses 20 evenly spaced frames per video.

The reported dataset processing produces:

- 40,000 frames
- 30,000 training frames
- 6,000 validation frames
- 4,000 test frames

The dataset is split at the video level.

Dataset source: https://www.kaggle.com/datasets/xdxd003/ffc23

## Environment requirements

Recommended environment:

- Python 3.x
- PyTorch
- torchvision
- OpenCV
- NumPy
- pandas
- scikit-learn
- Pillow
- dlib
- PyYAML
- matplotlib
- tqdm

GPU execution is recommended for training.

The reported experiments were executed within approximately:

- 6.5 GB system RAM
- 0.9 GB GPU VRAM
- NVIDIA T4 / Google Colab environment

## Installation

Clone the repository:

```bash
git clone https://github.com/Areshi001/Research-Assessment.git
cd Research-Assessment
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset preparation

Download the FaceForensics++ C23 dataset and place it in the configured dataset directory.

Update the dataset path in the configuration files under:

```text
configs/
```

The implementation expects the relevant original and DeepFakes video data to be available before frame extraction and region processing.

## Preprocessing

Frames are extracted using FFmpeg at 20 evenly spaced frames per video.

Full-face crops are obtained using a Haar cascade detector.

Facial landmarks are obtained using dlib.

The evaluated facial regions are:

* Periocular: landmarks 36–47
* Mouth: landmarks 48–67
* Cheeks: landmarks 1–5 and 11–15

Each region receives a 15-pixel expansion and is resized to:

```text
224 × 224
```

## Model configuration

### Full-face baseline

The baseline uses:

```text
MobileNetV3 Small
ImageNet pretrained weights
224 × 224 input
```

The model is fine-tuned for binary deepfake classification.

### Region-specific models

Three EfficientNet-B0 classifiers are trained independently:

```text
EfficientNet-B0 → Periocular
EfficientNet-B0 → Mouth
EfficientNet-B0 → Cheeks
```

The same general training configuration is used across the region-specific models.

### Temporal model

For the temporal comparison:

```text
MobileNetV3 Small
        ↓
Frozen feature extractor
        ↓
1024-dimensional frame feature
        ↓
8-frame sequence
        ↓
GRU
        ↓
Hidden size = 128
        ↓
Binary classification head
```

## Training configuration

The main training configuration uses:

```text
Optimizer: AdamW
Learning rate: 3e-4
Weight decay: 1e-4
Scheduler: OneCycleLR
Precision: FP16 mixed precision
Batch size: 64
Early stopping patience: 3
Selection criterion: Validation AUC
```

The temporal model uses sequences of:

```text
8 frames
Stride: 1
GRU hidden size: 128
```

## Training

Run the full-face baseline:

```bash
python src/train.py --config configs/baseline.yaml
```

Run the periocular experiment:

```bash
python src/train.py --config configs/periocular.yaml
```

Run the mouth experiment:

```bash
python src/train.py --config configs/mouth.yaml
```

Run the cheek experiment:

```bash
python src/train.py --config configs/cheeks.yaml
```

Run the temporal model:

```bash
python src/train.py --config configs/gru.yaml
```

## Evaluation

Evaluate a trained model using:

```bash
python src/evaluate.py --checkpoint <PATH_TO_CHECKPOINT> --config <PATH_TO_CONFIG>
```

The primary evaluation metric is:

```text
AUC-ROC
```

Additional metrics:

```text
Accuracy
Precision
Recall
```

Peak system RAM and GPU VRAM should also be recorded where possible because computational efficiency is part of the research question.

## Preliminary results

The currently reported test results on FF++ C23 are:

| Model                         | Accuracy | Precision | Recall |    AUC |
| ----------------------------- | -------: | --------: | -----: | -----: |
| Full Face (MobileNetV3 Small) |   0.6925 |    0.6831 | 0.7012 | 0.7872 |
| Mouth (EfficientNet-B0)       |   0.7573 |    0.7402 | 0.7504 | 0.8453 |
| Cheeks (EfficientNet-B0)      |   0.7627 |    0.7453 | 0.7572 | 0.8558 |
| GRU Temporal                  |   0.7600 |    0.7460 | 0.7870 | 0.8654 |
| Periocular (EfficientNet-B0)  |   0.7731 |    0.7482 | 0.7846 | 0.8681 |

These are measured experimental results rather than projected outcomes.

The periocular model achieved the highest reported AUC of:

```text
0.8681
```

The full-face MobileNetV3 Small baseline achieved:

```text
0.7872
```

The temporal GRU achieved:

```text
0.8654
```

## Outputs

Generated experiment outputs should be stored under:

```text
outputs/
```

Suggested organization:

```text
outputs/
├── figures/
├── results/
└── sample_predictions/
```

Examples of outputs include:

* Training curves
* ROC curves
* Confusion matrices
* Region comparison plots
* Evaluation metrics
* Sample predictions
* Resource usage measurements

## Reproducibility

To reproduce an experiment:

1. Install the dependencies.
2. Download and prepare FaceForensics++ C23.
3. Configure the dataset path.
4. Select the desired configuration from `configs/`.
5. Run the corresponding training command.
6. Evaluate the resulting checkpoint.
7. Inspect the generated files under `outputs/`.

All reported results must correspond to experiments that were actually executed.

Projected or expected results must not be presented as measured results.

## Proof of concept

The file `Deepfake_Detection_FINAL.ipynb` is the original Colab notebook containing the complete pipeline implementation, including frame extraction, region segmentation, model training, and evaluation. It serves as a reproducible proof of concept for all reported results.

## Project scope

The implemented research component is the facial-region sensitivity comparison.

The evaluated regions are:

* Full face
* Periocular
* Mouth
* Cheeks

The temporal GRU is included as a secondary comparison.

The following extensions are outside the current implementation scope:

* C40 evaluation
* Raw/uncompressed video evaluation
* Face2Face
* FaceSwap
* NeuralTextures
* Periocular-specific temporal modeling

These are considered future extensions.

## Limitations

The current study focuses on:

* DeepFakes manipulation
* H.264 C23 compression
* A limited set of facial regions
* dlib-based facial landmark detection

Landmark detection can fail on heavily compressed or occluded frames.

The current implementation therefore does not claim robustness across all manipulation types, compression levels, or real-world conditions.

## Repository structure

```text
deepfake-region-sensitivity/
│
├── README.md
├── requirements.txt
├── Deepfake_Detection_FINAL.ipynb
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── extract_regions.py
│   └── utils.py
│
├── configs/
│   ├── baseline.yaml
│   ├── periocular.yaml
│   ├── mouth.yaml
│   ├── cheeks.yaml
│   └── gru.yaml
│
└── outputs/
    ├── figures/
    ├── results/
    └── sample_predictions/
```

## References

[1] A. Rossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Niessner, "FaceForensics++: Learning to detect manipulated facial images," IEEE/CVF ICCV, 2019.

[2] A. Howard et al., "Searching for MobileNetV3," IEEE/CVF ICCV, 2019.

[3] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," ICML, 2019.

[4] D. E. King, "Dlib-ml: A machine learning toolkit," Journal of Machine Learning Research, vol. 10, pp. 1755–1758, 2009.

[5] FaceForensics++ C23 Dataset: https://www.kaggle.com/datasets/xdxd003/ffc23

[6] "DeepFakes Detection Across Generations: Analysis of Facial Regions, Fusion, and Performance Evaluation," Engineering Applications of Artificial Intelligence, 2022. DOI: 10.1016/j.engappai.2022.104673

[7] "A Dynamic Ensemble Selection of Deepfake Detectors Specialized for Individual Face Parts," MDPI Electronics, 2023. DOI: 10.3390/electronics12183932

[8] "TSFF-Net: A Deep Fake Video Detection Model Based on Two-Stream Feature Domain Fusion," PLOS ONE, 2024. DOI: 10.1371/journal.pone.0311366

[9] M. S. Rana, M. N. Nobi, B. Murthy, and A. H. Sung, "LightFakeDetect: A lightweight model for deepfake detection in videos that focuses on facial regions," MDPI Mathematics, 2025. DOI: 10.3390/math13193088

[10] "Beyond Spatial Frequency: Pixel-wise Temporal Frequency-based Deepfake Video Detection," 2025. https://arxiv.org/abs/2507.02398

[11] "Lightweight Deepfake Detection Based on Multi-Feature Fusion," 2025. https://arxiv.org/abs/2502.11763

[12] "Real Time Detection of Deepfakes Using the Efficient Swin Attention Network with Global and Local Facial Features," Discover Artificial Intelligence, 2026. DOI: 10.1007/s44163-026-01188-1

[13] "Robust Deepfake Detection in Compressed Videos with Scalable Network Strategies," Expert Systems with Applications, 2026. DOI: 10.1016/j.eswa.2026.131761
