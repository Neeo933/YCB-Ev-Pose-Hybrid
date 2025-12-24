这是一个为你定制的 `README.md` 文件。它结合了你之前的代码逻辑、论文亮点以及具体的复现步骤。你可以直接复制到你的 GitHub 仓库根目录下。

---

# GMG-PVNet: 面向复杂环境与高速运动场景的事件驱动6D物体姿态估计

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

**项目主页**: [https://github.com/Neeo933/YCB-Ev-Pose-Hybrid](https://github.com/Neeo933/YCB-Ev-Pose-Hybrid)

## 📖 项目简介 (Introduction)

**GMG-PVNet** (Geometry-Guided Multi-modal Network) 是一种针对极端视觉场景（强光、高速运动、弱纹理）设计的 6D 物体姿态估计网络。

传统 RGB-D 方法在运动模糊或光照剧烈变化下容易失效。本项目创新性地引入了 **事件相机 (Event Camera)** 数据，通过以下核心技术解决了上述挑战：

1.  **MTS (Motion-Texture Surface)**: 将高频异步事件流编码为 2D 纹理图像，捕捉不受光照影响的运动边缘。
2.  **AFDM (Attentive Feature Diffusion Module)**: 利用稀疏的事件点云引导密集图像特征的增强，解决跨模态融合难题。
3.  **Gated Fusion**: 自适应门控融合机制，根据信噪比动态加权纹理（RGB）与几何（Depth/Event）特征。
4.  **鲁棒训练策略**: 引入 Otsu 动态掩码生成与课程学习策略，在弱监督下实现高精度收敛。

在 YCB-Video / YCB-Ev 数据集上，本方法在对称物体上实现了 **>90% ADD-S** 的精度，并显著降低了姿态估计的平均误差。

---

## 🛠️ 环境依赖 (Prerequisites)

请确保安装以下依赖库：

```bash
conda create -n gmg_pose python=3.8
conda activate gmg_pose
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy tqdm scipy matplotlib
```

---

## 📂 数据准备 (Data Preparation)

### 1. 下载数据集
本项目基于 **YCB-Video (PBR)** 和 **YCB-Ev** 数据集。请下载数据并按照以下结构组织：

```text
../dataset/
├── test_pbr/                  # 数据集根目录
│   ├── 000001/                # 场景 ID
│   │   ├── rgb/               # RGB 图像
│   │   ├── depth/             # 深度图 (16-bit PNG)
│   │   ├── rgb_events/        # 事件累积图 (MTS 源数据)
│   │   ├── scene_camera.json  # 相机内参
│   │   ├── scene_gt.json      # 6D 姿态真值
│   │   └── scene_gt_info.json # BBox 信息
│   ├── ...
```

### 2. 数据预处理与 MTS 生成
运行数据工厂脚本，生成训练所需的 **关键点标签 (.npy)**、**MTS 裁剪图** 和 **Template 模板**。

```bash
python data_factory.py
```

*   **功能**：
    *   读取原始 RGB 和 Event 数据。
    *   根据物体 BBox 裁剪并生成 MTS (Motion-Texture Surface) 图像。
    *   将 3D 边界框角点投影到 2D，生成 `.npy` 格式的关键点真值。
    *   筛选高质量样本 (`visib > 0.8`) 生成用于匹配的 Template 库。
*   **输出**：生成的数据将保存在 `./processed_data` 目录下。

---

## 🚀 训练 (Training)

我们支持 **单物体训练** (快速验证) 和 **全物体训练**。训练脚本会自动执行课程学习策略（前 5 Epoch 学习分割，后 45 Epoch 学习向量场）。

### 运行训练

```bash
python train.py
```

### 配置说明 (`train.py` 中的 `CONFIG`)

你可以通过修改 `train.py` 顶部的 `CONFIG` 字典来调整实验设置：

*   **消融实验开关**:
    ```python
    "use_event_points": True,   # 是否启用 PointNet 分支 (Full Model)
    "ablation": {
        "use_geo": True,        # 是否使用几何分支 (Depth/MTS)
        "use_gate": True,       # 是否使用门控融合
        "use_afdm": True        # 是否使用 AFDM
    }
    ```
*   **训练模式**:
    *   **单物体**: 设置 `"target_obj_id": 3` (例如训练饼干盒)。
    *   **全物体**: 设置 `"target_obj_id": None`。

训练日志和模型权重将保存在 `./cloudcheckpoint/{exp_name}` 目录下。

---

## 📊 评估与推理 (Evaluation)

使用 `benchmark.py` 对训练好的模型进行定量评估。该脚本集成了 **RANSAC 投票**、**深度修正 (Depth Refinement)** 和 **ADD/ADD-S 指标计算**。

```bash
python benchmark.py
```

### 评估功能
1.  **定量指标**: 输出 `<2cm`, `<5cm`, `<10% diameter` 的准确率以及平均误差 (mm)。
2.  **定性可视化**: 在 `./benchmark_vis` 目录下生成对比图（左图为 GT，右图为预测结果，包含 3D 绿框和坐标轴）。

**示例输出**:
```text
Model: ./cloudcheckpoint/with_points/best.pth
--------------------
✅ Acc (< 2 cm):  25.8%
✅ Acc (< 5 cm):  67.0%
🏆 AUC (0-10cm):  30.8%
📏 Mean ADD Error: 23.0 mm
--------------------
```

---

## 🧩 核心代码结构

*   `data_factory.py`: 数据预处理、标签生成、MTS 生成。
*   `c2dataset.py`: PyTorch Dataset 定义，包含 **Otsu 动态掩码生成** 和 **在线点云采样**。
*   `c3model.py`: **GMG-PVNet** 模型定义，包含 FeatureExtractor, GatedFusion, AFDM, PointNet。
*   `c4train.py`: 训练主循环，包含混合精度训练和 Loss 记录。
*   `a5loss.py`: 复合损失函数 (Segmentation + Vector Field)。
*   `a6benchmark.py`: 推理流水线，包含 RANSAC Voting 和 PnP 解算。

---

## 📝 引用

如果您觉得本项目对您的研究有帮助，请引用：

```bibtex
@article{nan2025gmgpvnet,
  title={GMG-PVNet: Geometry-Guided Multi-modal Network for 6D Object Pose Estimation},
  author={Nan, Xiaolu and Xu, Jiajie},
  journal={arXiv preprint},
  year={2025}
}
```

---

**如有问题，欢迎提 Issue 或联系作者。**