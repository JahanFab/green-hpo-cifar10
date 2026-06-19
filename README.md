# Green Hyperparameter Optimization on CIFAR-10

Comparing **Grid Search** and **Random Search** for CNN hyperparameter tuning on CIFAR-10, measuring not just **accuracy** but also the **CO₂ emissions** of each search strategy.

The core question: *can Random Search match Grid Search's accuracy at a fraction of the carbon cost?*

## Overview

This notebook runs a controlled hyperparameter search across three pretrained backbones and benchmarks Grid Search against Random Search on two axes that usually get studied separately: predictive performance and environmental cost.

- **Dataset:** CIFAR-10 (10 classes, images resized to 128×128)
- **Models:** EfficientNetB0, ResNet50, MobileNetV2 (ImageNet-pretrained, fully fine-tuned with custom classification heads)
- **Search methods:** Grid Search (exhaustive over a fixed grid) vs Random Search (sampled from a wider value pool)
- **Metrics:** test accuracy, F1 / precision / recall, ROC-AUC, training time, and CO₂ emissions (grams) per trial and cumulative

## Hyperparameter search space

| Parameter | Values searched |
|---|---|
| Learning rate | 1e-3, 5e-4, 2e-4, 1e-4 |
| Dropout rate | 0.2 – 0.4 |
| Dense units | 64, 128, 256 |
| Optimizer | Adam, SGD |

Grid Search sweeps the full combination set; Random Search samples a smaller number of trials from a finer-grained pool so the two methods genuinely explore different points in the space.

## Key implementation details

- **GPU-cached data in float16** — the full dataset is resized, normalized, and stored in VRAM to remove DataLoader overhead during training.
- **Mixed-precision training** with `torch.autocast` and gradient scaling.
- **Early stopping** (patience-based) with best-model-state saving to avoid redundant retraining.
- **Emissions tracking** to attribute carbon cost to each individual trial and to the best configuration found.

## Results produced

Running the notebook generates:

- `summary_results.csv` — per-model, per-method summary
- `full_results.json` — complete trial-level results
- Comparison figures (accuracy, time, CO₂, and trial curves)

## Requirements

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn codecarbon
```

A CUDA-capable GPU is strongly recommended (the notebook caches the dataset to GPU memory).

## Usage

Open the notebook and run the cells top to bottom:

```bash
jupyter notebook Cifar10_latest.ipynb
```

Outputs are written to `./cifar10_output/`.

## Motivation

Hyperparameter search is one of the most compute-intensive (and therefore carbon-intensive) parts of an ML workflow. This experiment is part of broader work on **green machine learning**, measuring whether smarter search strategies can cut emissions without sacrificing accuracy.
