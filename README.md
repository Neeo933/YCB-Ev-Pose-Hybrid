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
├── 0attachment/              # Attachment files
├── 1model_script/            # Model training scripts
│   ├── baseline.py           # ResNet-18 baseline training script
│   ├── baseline_modified.py  # Modified baseline training script
│   ├── get_data_log.py       # Data logging script
│   ├── get_test_data.py      # Test data extraction script
│   ├── train_hybrid.py       # Hybrid model training script
│   ├── hybrid_pose_transformer.pth     # Trained hybrid model weights
│   ├── resnet18_pose_baseline.pth      # Trained ResNet-18 baseline weights
│   └── resnet18_pose_baseline_v3.pth   # Updated ResNet-18 baseline weights
├── 2visualize/               # Visualization tools
│   ├──
├── 3decode_data/             # Data decoding utilities
│   ├── generate_slice3.py    # Slice generation script
│   └── get_rawdata.py        # Raw data extraction script
├── ycb_ev_data/              # Dataset folder
│   ├── dataset/
│   │   └── test_pbr/        # Test data
│   └── test_pbr.zip          # Compressed test data
├── README.md                 # This file
└── result.md                 # Results documentation
```

## 📊 Results
| Model                | Train Loss | Val Loss | Analysis                           |
|----------------------|------------|----------|------------------------------------|
| ResNet-18 (Baseline) | 0.016      | 0.155    | Strong baseline, severe overfitting|
| Hybrid Transformer   | 0.013      | 0.160    | Better fitting capacity, needs more data |

## 🚀 Progress & Status

| File Name | Description | Status |
| :--- | :--- | :--- |
| `baseline.py` | Initial ResNet-18 training script. | Completed |
| `train_hybrid.py` | Script for training the fusion model. | Ongoing |
| **`baseline_modified.py`** | **Optimized ResNet-18 model with new augmentation/hyperparameters.** | **🥇 Best Model/ Ongoing** |
| `visualize.py` | Tools for visualizing results and feature maps. | Aborted |

## 🛠️ Usage
Download dataset:
```python get_data_log.py```

Train baseline:```
python baseline.py```

## 👨‍💻 Author
- **Neeo**, the Lead Developer.
- **Lumin**, the Company Founder.

## 👨‍💻 Contributor
- **Fn**, in charge of Publicity.