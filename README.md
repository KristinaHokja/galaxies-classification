# 🌌 Galaxy Morphology Classification

A machine learning project for automatic classification of galaxy morphologies from images, developed as part of the *Machine Learning* course (MSc, University of Pavia, 2025).

Two classification approaches are implemented and compared: a **Logistic Regression** model with handcrafted features and a **Convolutional Neural Network** based on transfer learning with ResNet50.

---

## Problem Definition

Galaxies come in four morphological categories:

| Class | Description |
|-------|-------------|
| **Spiral** | Disc-shaped with spiral arms |
| **Barred Spiral** | Spiral with a central bar structure |
| **Elliptical** | Smooth, featureless ellipsoidal shape |
| **Irregular** | No defined structure |

**Dataset:** 11,000 images (224×224 px, RGB), balanced across 4 classes — 10,000 training / 1,000 test.

**Special challenges addressed:**
- How much does **color information** help classification?
- How to exploit **rotation invariance** (galaxies have no preferred orientation)?

---

## Approaches

### 1. Logistic Regression with Feature Engineering (`Galaxies_classificationLR_HOKJA.ipynb`)

A linear classifier trained on handcrafted image features:

- **Grayscale histogram** (64 bins) — intensity distribution
- **Edge direction histogram** (Sobel, 64 bins) — structural features
- **Co-occurrence matrix** (64 features) — texture patterns
- **Color histogram** (RGB, 64 bins) — color information

Multiple feature combinations are benchmarked to isolate the contribution of each feature type.

**Key findings:**
- Best configuration: Shape + Texture features (`Shape_Texture`)
- Color adds limited benefit for morphological classification
- Dataset is balanced → accuracy is a reliable metric

### 2. CNN with Transfer Learning (`Galaxies_classificationCNN_HOKJA.ipynb`)

Fine-tuning of a **ResNet50** pre-trained on ImageNet, with the final fully-connected layer replaced for 4-class classification.

Three experimental configurations:

| Configuration | Input | Augmentation |
|--------------|-------|--------------|
| **RGB** | 3-channel color | Standard |
| **Grayscale (baseline)** | 1-channel | Standard |
| **Gray + RotInv** | 1-channel | Rotation-invariant augmentation |

Rotation-invariant augmentation (`RandomRotation(360°)`) exploits the physical symmetry of galaxy images to improve generalisation.

---

## Repository Structure

```
galaxies-classification/
│
├── Galaxies_classificationLR_HOKJA.ipynb   # Logistic Regression notebook
├── Galaxies_classificationCNN_HOKJA.ipynb  # CNN / ResNet50 notebook
│
├── galaxies/
│   ├── train/
│   │   ├── spiral/
│   │   ├── barred/
│   │   ├── elliptical/
│   │   └── irregular/
│   └── test/
│       ├── spiral/
│       ├── barred/
│       ├── elliptical/
│       └── irregular/
│
└── Galaxies_classification.pdf   # Assignment specification
```

---

## Requirements

```bash
pip install torch torchvision numpy matplotlib opencv-python scikit-learn gdown tqdm
```

Notebooks are designed to run on **Google Colab** with GPU acceleration. The dataset is downloaded automatically via `gdown` from Google Drive.

---

## Usage

Open the notebooks in Google Colab:

**Logistic Regression:**
```
Galaxies_classificationLR_HOKJA.ipynb
```
Run all cells sequentially. The notebook covers data exploration, feature extraction, model training, confusion matrix analysis, and per-class error inspection.

**CNN (ResNet50):**
```
Galaxies_classificationCNN_HOKJA.ipynb
```
Run all cells sequentially. Three experiments are run: RGB, Grayscale, and Grayscale + rotation-invariant augmentation. Confusion matrices are generated for each.

---

## Results Summary

| Model | Configuration | Test Accuracy |
|-------|--------------|---------------|
| Logistic Regression | Shape + Texture | see notebook |
| ResNet50 | RGB | see notebook |
| ResNet50 | Grayscale | see notebook |
| ResNet50 | Gray + RotInv | see notebook |

*Full metrics and confusion matrices are available in the notebooks.*

---

## References

- He et al. — *Deep Residual Learning for Image Recognition* (ResNet, 2016)
- Galaxy Zoo project — [www.galaxyzoo.org](https://www.galaxyzoo.org)

---

*BSc Physics — Machine Learning Course, University of Pavia, 2025*  
*Author: Kristina Hokja*
