import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from pathlib import Path
from collections import Counter

# 假设你的 Dataset 类保存在 dataset.py 中
from a2dataset import GMGPoseDataset 

def draw_xyz_axis(img, kpts, axis_len=None):
    """
    在图像上画出 XYZ 坐标轴
    img: cv2 图像
    kpts: 9个关键点坐标 (前面8个是角点，第9个是中心点)
    """
    img = img.copy()
    
    # 1. 拿到中心点 (Point 8)
    cx, cy = kpts[8]
    
    # 2. 计算三个轴的“虚拟端点”
    # X轴正向: 点 0,1,2,3 的中心
    x_tip = kpts[0:4].mean(axis=0)
    
    # Y轴正向: 点 0,1,4,5 的中心
    y_indices = [0, 1, 4, 5] 
    y_tip = kpts[y_indices].mean(axis=0)
    
    # Z轴正向: 点 0,2,4,6 的中心
    z_indices = [0, 2, 4, 6]
    z_tip = kpts[z_indices].mean(axis=0)
    
    # 3. 绘制直线 (OpenCV 是 BGR 顺序)
    # 画 Z 轴 (蓝色 Blue)
    cv2.arrowedLine(img, (int(cx), int(cy)), (int(z_tip[0]), int(z_tip[1])), 
                    (255, 0, 0), 2, tipLength=0.2)
    
    # 画 Y 轴 (绿色 Green)
    cv2.arrowedLine(img, (int(cx), int(cy)), (int(y_tip[0]), int(y_tip[1])), 
                    (0, 255, 0), 2, tipLength=0.2)
    
    # 画 X 轴 (红色 Red)
    cv2.arrowedLine(img, (int(cx), int(cy)), (int(x_tip[0]), int(x_tip[1])), 
                    (0, 0, 255), 2, tipLength=0.2)
    
    # 在轴边上标字
    cv2.putText(img, "X", (int(x_tip[0]), int(x_tip[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, "Y", (int(y_tip[0]), int(y_tip[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, "Z", (int(z_tip[0]), int(z_tip[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return img

def test_gmg_dataset():
    # ================= 配置路径 =================
    PROCESSED_DIR = "./processed_data"
    DATASET_ROOT = "../ycb_ev_data/dataset/test_pbr"
    
    print(f"正在初始化 Dataset...")
    try:
        dataset = GMGPoseDataset(
            processed_dir=PROCESSED_DIR, 
            dataset_root=DATASET_ROOT,
            target_size=(128, 128),
            mode='train'
        )
    except Exception as e:
        print(f"Dataset 初始化失败: {e}")
        return

    # 统计 obj_id 分布
    all_obj_ids = [s['obj_id'] for s in dataset.sample_list]
    counts = Counter(all_obj_ids)

    print("\n=== 各物体样本数量统计 ===")
    for obj_id, count in sorted(counts.items()):
        print(f"Obj ID {obj_id}: {count} 个样本")

    # ================= 随机抽取 3 个样本进行测试 =================
    num_samples = 3
    if len(dataset) < num_samples:
        num_samples = len(dataset)
    indices = random.sample(range(len(dataset)), num_samples)

    for i, idx in enumerate(indices):
        print(f"\n>>> 测试样本 [{i+1}/{num_samples}] (Index: {idx})")
        
        sample = dataset[idx]
        
        # 1. 解包数据
        input_tensor = sample['input']         # [4, 128, 128]
        target_field = sample['target_field']  # [18, 128, 128]
        obj_id = sample['obj_id'].item()       # Scalar
        kpts_local = sample['kpts_local']      # [9, 2]
        depth_tensor = sample['depth']         # [1, 128, 128] <--- [新增] 获取深度图
        
        # 2. 控制台输出维度检查
        print(f"  - Obj ID: {obj_id}")
        print(f"  - Input Shape: {input_tensor.shape} (Exp: [4, 128, 128])")
        print(f"  - Depth Shape: {depth_tensor.shape} (Exp: [1, 128, 128])") # <--- [新增] 检查
        print(f"  - Field Shape: {target_field.shape} (Exp: [18, 128, 128])")
        
        # 3. 数据还原 (用于可视化)
        # RGB
        rgb_vis = input_tensor[:3, :, :].numpy().transpose(1, 2, 0)
        rgb_vis = (rgb_vis * 255).astype(np.uint8)
        rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_BGR2RGB)

        # MTS
        mts_vis = input_tensor[3, :, :].numpy()
        mts_vis = (mts_vis * 255).astype(np.uint8)

        # Depth <--- [新增]
        # 深度图本身是 float (米单位)，直接可视化可能看不清，交给 matplotlib 的 cmap 处理
        depth_vis = depth_tensor[0, :, :].numpy() 

        # 4. 可视化长方体和坐标轴
        overlay_img = rgb_vis.copy()
        kpts_np = kpts_local.numpy()

        # 画坐标轴
        overlay_img = draw_xyz_axis(overlay_img, kpts_np)

        # 定义立方体的12条边
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (3, 7), (2, 6)
        ]
        for start_idx, end_idx in edges:
            start_point = tuple(map(int, kpts_np[start_idx]))
            end_point = tuple(map(int, kpts_np[end_idx]))
            cv2.line(overlay_img, start_point, end_point, (0, 255, 0), 1)
        
        for kp_i, (x, y) in enumerate(kpts_np):
            color = (0, 255, 0) if kp_i < 8 else (255, 0, 0)
            cv2.circle(overlay_img, (int(x), int(y)), 2, color, -1)

        # 5. 画图 (修改为 4 列)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4)) # <--- 改为 1 行 4 列
        
        # 子图 1: RGB
        axes[0].imshow(rgb_vis)
        axes[0].set_title(f"RGB (ID:{obj_id})")
        axes[0].axis('off')
        
        # 子图 2: MTS
        axes[1].imshow(mts_vis, cmap='gray')
        axes[1].set_title("MTS (Event)")
        axes[1].axis('off')

        # 子图 3: Depth (新增)
        # 使用 'plasma' 或 'magma' 热力图显示深度，越亮代表值越大（越远），或反之
        im_depth = axes[2].imshow(depth_vis, cmap='plasma') 
        axes[2].set_title(f"Depth (m)\nMin:{depth_vis.min():.2f} Max:{depth_vis.max():.2f}")
        axes[2].axis('off')
        # 加个颜色条看数值对不对 (应该是 0.x 到 2.x 米左右)
        plt.colorbar(im_depth, ax=axes[2], fraction=0.046, pad=0.04)

        # 子图 4: Overlay
        axes[3].imshow(overlay_img)
        axes[3].set_title("GT Pose Overlay")
        axes[3].axis('off')
        
        plt.suptitle(f"Sample Index: {idx} | Check Depth Quality")
        plt.tight_layout()
        plt.show()

        # 6. 简单的数值检查
        cx, cy = int(kpts_np[8][0]), int(kpts_np[8][1])
        cx, cy = np.clip(cx, 0, 127), np.clip(cy, 0, 127)
        vec_x = target_field[16, cy, cx].item()
        vec_y = target_field[17, cy, cx].item()
        print(f"  - Vector Check: ({vec_x:.4f}, {vec_y:.4f})")

if __name__ == "__main__":
    test_gmg_dataset()