# Enhancing Lung Radiology Image Classification through Data Synthesis and Contrastive Learning Techniques

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Optional: Add a license badge -->

## Introduction

Deep learning models have shown great promise in medical image analysis, particularly for lung radiology (Chest X-Rays, CT Scans) which is crucial for diagnosing various pulmonary diseases. However, a major bottleneck is the scarcity of large, accurately annotated datasets due to the expensive and time-consuming nature of expert annotation.

This project tackles this data scarcity problem by exploring two key strategies:

1.  **Data Synthesis:** Generating realistic, class-conditioned lung radiology images using Generative Adversarial Networks (GANs), specifically Conditional GANs (**DCGAN**) and Auxiliary Classifier GANs (**ACGAN**), to augment the training data and potentially balance class distributions.
2.  **Self-Supervised Pretraining:** Employing contrastive learning techniques (**SimCLR**, **DINO**) to pretrain vision models (**ResNet**, **Vision Transformer**) on the image data (real and potentially synthetic) without relying on explicit labels. This aims to learn robust representations beneficial for downstream classification tasks.

The ultimate goal is to improve the classification performance on a lung radiology dataset by combining these techniques and comparing their effectiveness against baseline approaches.

## Dataset

*   **Source:** A subset of the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) available on Kaggle.
*   **Content:** Approximately 4000 Chest X-Ray images.
*   **Classes:**
    *   Normal
    *   COVID-19
    *   Lung Opacity (Non-COVID lung infection)
    *   Viral Pneumonia

## Methods Overview

1.  **Data Synthesis:** Train DCGAN and ACGAN models conditioned on class labels to generate synthetic images for the four specified classes. Quality will be assessed visually and using the Frechet Inception Distance (FID) score.
2.  **Contrastive Learning:** Pretrain ResNet and ViT backbones using SimCLR and DINO frameworks on the available lung radiography images (potentially augmented with synthetic data).
3.  **Methods**:
    *   **(Baseline model):** Fine-tune standard ImageNet-pretrained ResNet/ViT on the real dataset.
    *   **(Baseline model 2):** Train ResNet/ViT (from ImageNet weights) on the real dataset augmented with synthetic images.
    *   **(Proposed):** Fine-tune the contrastive-pretrained ResNet/ViT on the real, labeled dataset.
4.  **Evaluation:** Compare the classification performance of the different approaches using metrics like Accuracy, Weighted/Macro F1-Score, Precision, and Recall.

## Environment Setup

**1. Clone the Repository:**

```
git clone https://github.com/dvtiendat/contrastive-synthesis-medcls.git
cd contrastive-synthesis-medcls 
```

**2. Create and activate conda environment**
```
conda env create -f environment.yaml
conda activate medcls
```

For Kaggle usage: 
```
!pip install -r requirements.txt
```
