# 1218生成时间切片
## 3. Data Preparation: Raw Stream Decoding and Micro-Temporal Encoding

### 3.1 Reverse Engineering of Raw Event Streams
To overcome the temporal aliasing inherent in standard 2D event histograms, we directly process the raw binary streams (`.int32.zst`) provided by the YCB-Ev SD dataset. Through statistical analysis of the bitstream, we reverse-engineered the storage protocol, identifying a compact $(N, 2)$ layout comprising monotonically increasing timestamps $t$ and bit-packed spatial data $D_{packed}$. We resolved the bit-wise decoding logic as $D_{packed} = (p \ll 28) \ | \ (y \ll 14) \ | \ x$ and calibrated the spatial resolution to the VGA standard ($640 \times 480$). This allows us to reconstruct the lossless event point cloud $\mathcal{E} = \{x_i, y_i, t_i, p_i\}$ from compressed artifacts.

### 3.2 Micro-Temporal Slicing (MTS)
Leveraging the high temporal resolution of event cameras, we propose a **Micro-Temporal Slicing** strategy to explicitly encode motion dynamics into a static representation. The continuous event stream within a 34ms window is normalized and partitioned into three equidistant temporal bins. These bins are accumulated and mapped to the R, G, and B channels, respectively:
$$
I(u, v) = \bigcup_{c \in \{R, G, B\}} \sum_{e_k \in \mathcal{T}_c} \mathbb{1}(x_k=u, y_k=v)
$$
where $\mathcal{T}_R$ (Past), $\mathcal{T}_G$ (Present), and $\mathcal{T}_B$ (Future) represent the temporal partitions.

### 3.3 Benefits
This approach yields a **Chromatic-Temporal Representation**. Unlike grayscale time-surfaces which compress motion into intensity, our encoding translates motion direction and velocity into observable color gradients (e.g., a "Red-to-Blue" trail indicates trajectory). This provides the subsequent CNN with explicit geometric priors for 6DoF pose estimation without introducing additional computational overhead.

# 1220数据预处理 + 数据加载与监督信号生成

**Data Preprocessing.** We construct the training pairs by generating ground truth keypoints and reference templates from the dataset. For 6D pose supervision, we define a virtual 3D cuboid with a reference scale of 50mm in the object coordinate system. The ground truth 2D keypoints are obtained by projecting 9 control points (8 corners and 1 centroid) onto the image plane using the camera intrinsics $K$ and the ground truth pose $[R|t]$. To construct the support set for model-free matching, we filter instances with a visibility fraction greater than 0.8 to ensure visual integrity. These high-quality object crops (both RGB and MTS modalities) are extracted and resized to a fixed resolution of $128 \times 128$ pixels to serve as reference templates.

### 中文版 (适用于“实现细节”或“数据准备”章节)

**数据预处理：** 我们通过从原始数据集中生成真值关键点和参考模板来构建训练样本。为了进行6D姿态监督，我们在物体坐标系下定义了一个参考尺度为50mm的虚拟3D立方体。通过结合相机内参 $K$ 和真值姿态 $[R|t]$，我们将9个控制点（8个立方体角点和1个质心）投影至图像平面以获取2D真值关键点。为了构建用于无模型匹配的支持集（Support Set），我们筛选出可见度分数大于0.8的实例以确保外观的完整性。这些高质量的物体裁剪图（包含RGB和MTS模态）被提取并统一调整为 $128 \times 128$ 像素的分辨率，作为标准参考模板。



**Dataset Implementation and Supervision Generation.**
We implement a custom PyTorch dataset to facilitate robust training. To ensure high-quality supervision, we perform an online filtering strategy during the sample list construction: instances with a visibility fraction lower than 10% ($v < 0.1$) or invalid bounding boxes are discarded, as they lack sufficient visual features for pose regression.
During training, we apply a dynamic spatial transformation pipeline. A bounding box with a 15% padding ratio is applied to crop the target object from both RGB and MTS images, ensuring context preservation. The crops are resized to a fixed resolution of $128 \times 128$. Crucially, the ground truth keypoints are transformed from the global image coordinate system to the local cropped coordinate system. Finally, we generate the pixel-wise unit vector field $V \in \mathbb{R}^{H \times W \times 18}$ on-the-fly. For each pixel $p$, the vector field represents the unit direction pointing to the $k$-th keypoint, serving as the dense voting target for the network. The input to the network is a concatenated 4-channel tensor (RGB + MTS).

---

### 中文版 (适用于“方法论”或“实现细节”)

**数据集实现与监督信号生成：**
我们实现了一个定制的 PyTorch 数据集以支持鲁棒的训练过程。为了确保监督信号的质量，我们在构建样本列表时采用了在线过滤策略：可见度分数低于 10% ($v < 0.1$) 或包围盒无效的实例将被剔除，因为它们缺乏足够的视觉特征用于姿态回归。
在训练过程中，我们应用了动态空间变换流水线。我们使用具有 15% 填充率（Padding Ratio）的包围盒从 RGB 和 MTS 图像中裁剪目标物体，以确保边缘上下文的保留。裁剪后的图像被调整为固定的 $128 \times 128$ 分辨率。至关重要的是，真值关键点从全局图像坐标系被变换到了局部裁剪坐标系下。最后，我们在线（On-the-fly）生成像素级单位向量场 $V \in \mathbb{R}^{H \times W \times 18}$。对于每个像素 $p$，向量场表示指向第 $k$ 个关键点的单位方向，这作为网络密集投票（Dense Voting）的回归目标。网络的输入为拼接后的 4 通道张量（RGB + MTS）。

---

### 这段代码的 3 个亮点（写论文时心中的“潜台词”）

1.  **Quality Control (质量控制)**:
    *   代码：`if visib_fract < 0.1: continue`
    *   意义：你没有盲目地把所有数据喂给网络，而是像老师一样挑出了“好题”。这解释了为什么你的模型收敛更快、更稳。

2.  **Context Awareness (上下文感知)**:
    *   代码：`_pad_bbox(..., padding_ratio=0.15)`
    *   意义：你没有贴着物体边缘剪裁，而是留了一点余地。这对于处理物体边缘的特征提取至关重要，防止了 CNN 在边缘处的 Padding 效应导致信息丢失。

3.  **Efficiency (计算效率)**:
    *   代码：`_generate_vector_field` 在 `__getitem__` 里运行。
    *   意义：向量场数据体积很大（18通道 float32），如果存硬盘会把磁盘撑爆且读取慢。你在读取时由 CPU 实时计算，这是典型的 **"Compute over IO"（以计算换 IO）** 优化策略，非常专业。