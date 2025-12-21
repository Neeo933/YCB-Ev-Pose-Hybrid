import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def verify_mts_depth_alignment(mts_path, depth_path):
    # 1. 读取 MTS 图 (3通道彩色: R-过去, G-现在, B-未来)
    mts_img = cv2.imread(mts_path)
    if mts_img is None:
        print(f"Error: 找不到 MTS 文件 {mts_path}")
        return

    # 2. 读取深度图
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        print(f"Error: 找不到深度图 {depth_path}")
        return

    # 3. 深度图归一化
    depth_valid = depth_img.astype(float)
    depth_norm = cv2.normalize(depth_valid, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 辅助函数：计算梯度
    def get_gradient(img):
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad = cv2.magnitude(grad_x, grad_y)
        return cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- 改进点 1: 处理 MTS 三条边重影 ---
    # 分通道计算梯度，取最大值融合，减少因时间差导致的边缘加宽
    b, g, r = cv2.split(mts_img)
    grad_b = get_gradient(b)
    grad_g = get_gradient(g)
    grad_r = get_gradient(r)
    # 取三帧中响应最强的一个，消除重影
    grad_mts_raw = np.maximum(grad_b, np.maximum(grad_g, grad_r))

    # --- 改进点 2: 利用深度空间先验过滤背景纹理 ---
    grad_depth = get_gradient(depth_norm)
    
    
    # 创建掩码：只有深度发生突变的地方（物体边界），才是我们要修正的地方
    # 阈值可以根据实验调整，这里设为 20。膨胀 kernel 扩大搜索范围
    _, mask_edge = cv2.threshold(grad_depth, 20, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8) # 7x7 窗口足以覆盖 MTS 运动偏差
    spatial_mask = cv2.dilate(mask_edge, kernel)
    
    # 过滤：剔除那些在深度图上“一马平川”的背景纹理事件
    grad_mts_filtered = cv2.bitwise_and(grad_mts_raw, grad_mts_raw, mask=spatial_mask)

    # 5. 叠显对比 (红色: 过滤后的 MTS 边缘, 绿色: 原始 Depth 边缘)
    overlay = np.zeros_like(mts_img)
    overlay[:, :, 0] = grad_mts_filtered   # R通道
    overlay[:, :, 1] = grad_depth          # G通道

    # 6. 绘图展示
    plt.figure(figsize=(20, 12))
    
    plt.subplot(2, 3, 1)
    plt.title("Original MTS (3-way Ghosting)")
    plt.imshow(cv2.cvtColor(mts_img, cv2.COLOR_BGR2RGB))
    
    plt.subplot(2, 3, 2)
    plt.title("Filtered MTS Edge (Anti-Ghosting & Denoised)")
    plt.imshow(grad_mts_filtered, cmap='hot')

    plt.subplot(2, 3, 3)
    plt.title("Depth Spatial Prior (Mask)")
    plt.imshow(spatial_mask, cmap='gray')

    # plt.title("Depth Map (Spatial Prior)")
    # plt.imshow(depth_img, cmap='jet')

    plt.subplot(2, 3, 4)
    plt.title("Depth Gradient (Blurry)")
    plt.imshow(grad_depth, cmap='hot')

    plt.subplot(2, 3, 5)
    plt.title("Guided Alignment (Red:Event, Green:Depth)")
    plt.imshow(overlay)
    
    plt.subplot(2, 3, 6)
    # 最终引导效果预览：用 MTS 边缘强行锐化 Depth 特征
    refined_preview = cv2.addWeighted(depth_norm, 0.6, grad_mts_filtered, 0.8, 0)
    plt.title("Guided-6D Refinement Result")
    plt.imshow(refined_preview, cmap='magma')

    plt.tight_layout()
    plt.savefig("guided_6d_refined_motivation.png", dpi=300)
    plt.show()

# 运行验证
path_to_depth = "../ycb_ev_data/dataset/test_pbr/000000/depth" 
path_to_mts = "../ycb_ev_data/dataset/test_pbr/000000/rgb_events" # 请确认文件夹名
verify_mts_depth_alignment(f"{path_to_mts}/000001.png", f"{path_to_depth}/000001.png")