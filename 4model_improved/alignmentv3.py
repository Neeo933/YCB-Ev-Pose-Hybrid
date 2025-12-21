import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_academic_visualization(mts_path, depth_path, save_name="guided_6d_convincing_results.png"):
    # 1. 数据加载与预处理
    mts_img = cv2.imread(mts_path)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if mts_img is None or depth_raw is None:
        print("路径错误，请检查文件。")
        return

    h, w = depth_raw.shape
    depth_f = depth_raw.astype(np.float32)

    # 2. MTS 边缘锐化 (使用共识加权去重影)
    b, g, r = cv2.split(mts_img)
    def get_grad(channel):
        gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    # 通过权重分配 (Green为锚点) 解决“一个盒子三条边”的重影
    grad_refined_raw = get_grad(g) * 0.6 + get_grad(r) * 0.2 + get_grad(b) * 0.2
    grad_refined_raw = cv2.normalize(grad_refined_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. 自适应掩码 (解决背景纹理噪声)
    valid_depths = depth_f[depth_raw > 0]
    foreground_mask = np.zeros((h, w), dtype=np.uint8)
    if len(valid_depths) > 0:
        d_min = np.percentile(valid_depths, 1)
        d_max = np.percentile(valid_depths, 99)
        # 动态弹性窗口
        far_p = d_min + max(0, (d_max - d_min) * 0.9)
        binary_mask = ((depth_f >= d_min) & (depth_f <= far_p)).astype(np.uint8) * 255
        # 膨胀处理保护边缘
        binary_mask = cv2.dilate(binary_mask, np.ones((7, 7), np.uint8))
        # 连通域过滤：剔除桌面
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
        for i in range(1, num_labels):
            if 200 < stats[i, cv2.CC_STAT_AREA] < (h * w * 0.8):
                foreground_mask[labels == i] = 255

    # 4. 特征融合与对比
    mts_grad_final = cv2.bitwise_and(grad_refined_raw, grad_refined_raw, mask=foreground_mask)
    depth_display = cv2.normalize(depth_f, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_grad = get_grad(depth_display).astype(np.uint8)
    
    # 提取 MTS 独有的内部棱线
    internal_edges = cv2.subtract(mts_grad_final, depth_grad)
    _, internal_ridges = cv2.threshold(internal_edges, 35, 255, cv2.THRESH_BINARY)

    # --- 5. 绘制具有说服力的六张图 ---
    fig = plt.figure(figsize=(24, 14))
    plt.rcParams['font.size'] = 14

    # 图1: 原始输入 (MTS Chromatic)
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_title("1. Raw MTS (Input)\nChromatic-Temporal Encoding", color='blue', fontweight='bold')
    ax1.imshow(cv2.cvtColor(mts_img, cv2.COLOR_BGR2RGB))
    ax1.axis('off')

    # 图2: 深度图缺陷 (Depth Prior)
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_title("2. Depth Map Gradient\nBlurry Edges & Missing Ridges", color='red', fontweight='bold')
    ax2.imshow(depth_grad, cmap='hot')
    ax2.axis('off')

    # 图3: 自适应掩码过程 (Spatial Gating)
    # 展示我们如何干净地剥离背景纹理，只留下物体
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title("3. Adaptive Spatial Gating\nDenoising Background Textures", color='green', fontweight='bold')
    # 叠加显示：原图+Mask
    mask_overlay = cv2.addWeighted(cv2.cvtColor(depth_display, cv2.COLOR_GRAY2RGB), 0.5, 
                                   cv2.applyColorMap(foreground_mask, cv2.COLORMAP_JET), 0.5, 0)
    ax3.imshow(mask_overlay)
    ax3.axis('off')

    # 图4: 去重影后的锐化边缘 (Temporal Consensus)
    # 展示从“三条重影”到“一根细线”的进化
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title("4. Sharp MTS Edges\nAnti-Ghosting & Refined", color='purple', fontweight='bold')
    ax4.imshow(mts_grad_final, cmap='magma')
    ax4.axis('off')

    # 图5: 几何补全效果 (Geometric Completion) - 最关键的一张
    # 红色是深度图完全看不见的内部几何特征
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title("5. Geometric Completion\nGreen: Depth, Red: MTS Intra-Edge", color='darkorange', fontweight='bold')
    comparison = np.zeros((h, w, 3), dtype=np.uint8)
    comparison[:, :, 1] = depth_grad       # 绿色：深度跳变
    comparison[:, :, 2] = internal_ridges  # 红色：补全的内部棱线
    ax5.imshow(comparison)
    ax5.axis('off')

    # 图6: 最终引导特征表示 (Final Feature)
    # 这种图最能体现“Guided-6D”的审美
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_title("6. Final Guided Representation\nReady for Pose Estimation CNN", color='black', fontweight='bold')
    final_rep = cv2.addWeighted(depth_display, 0.4, mts_grad_final, 0.9, 0)
    ax6.imshow(final_rep, cmap='inferno')
    ax6.axis('off')

    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"学术展示图已生成: {save_name}")
    plt.show()

# 运行代码
D_PATH = "../ycb_ev_data/dataset/test_pbr/000000/depth/000005.png"
M_PATH = "../ycb_ev_data/dataset/test_pbr/000000/rgb_events/000005.png"
generate_academic_visualization(M_PATH, D_PATH)