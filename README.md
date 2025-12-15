# YCB-Ev-Pose-Hybrid: Event-based 6DoF Pose Estimation
> 2025 Autumn Computer Vision Course Project

## 📖 Introduction
This project explores efficient 6DoF object pose estimation using event camera data (YCB-Ev SD dataset). We compare a standard **ResNet-18** baseline against a proposed **Hybrid CNN-Transformer** architecture to analyze performance in sparse data regimes.

## 🚀 Features
- **Data Efficiency**: Analyzed pose estimation on sparse event histograms.
- **Hybrid Architecture**: Implemented a ResNet + Transformer Encoder model to capture global dependencies.
- **Optimization**: Customized loss function (Translation + Rotation separation) to alleviate overfitting.

## 📂 Project Structure
```text
.
├── baseline.py       # ResNet-18 baseline training script
├── train_hybrid.py   # Hybrid model training script
├── visualize.py      # Visualization tools
└── README.md