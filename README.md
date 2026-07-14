# Tumor prediction and detection using ELLM models and hybrid deep learning: Master Thesis Project

## Overview
Brain tumor segmentation and explainable reporting on the BraTS2021 dataset. The pipeline is split into 5 notebooks and app/ has the demo application built on top of them.

## Notebooks
phase1_preprocessing_experiments.ipynb: preprocessing the MRI scans

phase2_full_finetuning.ipynb: fine-tuning the segmentation model

phase3_feature_extraction.ipynb: extracting tumor features (size, shape, composition)

phase4_description_layer.ipynb: turning features into a readable findings report

phase5_explanaible_layer.ipynb:  explaining the findings with a local LLM and reference sources


Run them in order, each one needs the output of the previous one.

## App
app/scripts/ has the demo app:

brats_app.py: main app/UI

brats_inference.py: segmentation + feature extraction for one patient

report_figures.py: generates the report figures

rag_explainer.py: explanation module

brats_app_workflow.pdf shows a full run of the app on patient BraTS2021_00060.

What's in this repository:
```
NIKOLINA_TFM/
│
├── notebooks/
│   └── phase1_preprocessing_experiments.ipynb
│   └── phase2_full_finetuning.ipynb
│   └── phase3_feature_extraction.ipynb
│   └── phase4_description_layer.ipynb
│   └── phase5_explanaible_layer.ipynb
│
│
├── app/
│   └── scripts
│       └── brats_app.py
│       └── brats_inference.py
│       └── report_figures.py
│       └── rag_explainer.py
│   └── assets/
│       └── 05_report_features_final.json
│       └── brats21_folds.json
│       └── Params.yaml
│
├── requirements.txt
│
├── brats_app_workflow.pdf
│
└── README.md
```

What is missing to run it:
BraTS2021 dataset -> data/BraTS21/BraTS2021_Training_Data/TrainingData/

The official Synapse page for BraTS2021 (https://www.synapse.org/#!Synapse:syn27046444/wiki/616992) does no longer provide 2021-only dataset
Use the Kaggle platform instead: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1.

Fold split json file (brats21_folds.json). Download from: https://developer.download.nvidia.com/assets/Clara/monai/tutorials/brats21_folds.json.
Base pretrained model (starting point for phase2) -> pretrained_models/model_fold1/model.pt
Reference texts for the explanation layer (used by phase5 and rag_explainer.py) -> app/references/
Ollama installed locally with the llama3:8b model pulled for phase5 and rag_explainer.py

Full structure:
```
project/
├── data/
│   └── BraTS21/
│       └── BraTS2021_Training_Data/
│           └── TrainingData/
│ 
├── jsons/
│   └── brats21_folds.json
│
├── notebooks/
│   └── phase1_preprocessing_experiments.ipynb
│   └── phase2_full_finetuning.ipynb
│   └── phase3_feature_extraction.ipynb
│   └── phase4_description_layer.ipynb
│   └── phase5_explanaible_layer.ipynb
│
├── outputs/
│
├── pretrained_models/
│   └── model_fold1/
│       └── model.pt
│
├── app/
│   └── scripts
│       └── brats_app.py
│       └── brats_inference.py
│       └── report_figures.py
│       └── rag_explainer.py
│   └── assets/
│       └── 05_report_features_final.json
│       └── brats21_folds.json
│       └── Params.yaml
│       └── fine_tuning_best_model.pt
│   └── references/
│
├── requirements.txt
│
└── README.md
```

## Requirements
See "requirements.txt" for dependencies.

## References and acknowledgements

Phase 2 (segmentation fine-tuning) is adapted from MONAI's official Swin UNETR
BraTS21 tutorial:
https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb
(Copyright MONAI Consortium, Apache License 2.0). The fold-split json used for training/validation comes from that same tutorial.

Model:
[1] Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H. and Xu, D., 2022.
Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. arXiv:2201.01266.

[2] Tang, Y., Yang, D., Li, W., Roth, H.R., Landman, B., Xu, D., Nath, V. and
Hatamizadeh, A., 2022. Self-supervised pre-training of swin transformers for 3d medical image analysis. CVPR 2022, pp. 20730-20740.

Dataset:
[3] U. Baid, et al., The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification, arXiv:2107.02314, 2021.

[4] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)",
IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015).

[5] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert
segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017).

[6] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the
TCGA-GBM collection", The Cancer Imaging Archive, 2017.

This repository is licensed under MIT (see LICENSE), except for phase2_full_finetuning.ipynb, which is adapted from MONAI's Swin UNETR BraTS21 tutorial and remains subject to the Apache License 2.0 (Copyright MONAI Consortium).
