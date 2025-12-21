import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_guided_6d_edges_sharp(mts_path, depth_path, save_name="guided_6d_v6_sharp.png"):
    # 1. 加载
    mts_img = cv2.imread(mts_path)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if mts_img is None or depth_raw is None: return

    h, w = depth_raw.shape
    depth_f = depth_raw.astype(np.float32)

    # --- 2. 【核心改进】去重影边缘提取 (Anti-Ghosting Edge) ---
    b, g, r = cv2.split(mts_img)
    
    def get_sharp_grad(channel):
        gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    grad_b = get_sharp_grad(b)
    grad_g = get_sharp_grad(g)
    grad_r = get_sharp_grad(r)

    # 改进的融合逻辑：
    # 我们不只取最大值，而是通过“加权共识”来寻找边缘。
    # 这种方法会惩罚那些只在一个通道出现的离散噪声，并强化三个通道共同指向的边缘中心。
    # 权重分配：G通道(现在)占 50%，R和B(过去和未来)各占 25%
    combined_grad = grad_g * 0.5 + grad_r * 0.25 + grad_b * 0.25
    
    # 使用非极大值抑制（NMS）的思想：只保留局部梯度最大的点，让线条变细
    # 简单的形态学“瘦身”处理
    kernel_thin = np.ones((3,3), np.uint8)
    mts_grad_refined_raw = cv2.normalize(combined_grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 通过形态学开运算去掉细小的毛刺重影
    mts_grad_refined_raw = cv2.morphologyEx(mts_grad_refined_raw, cv2.MORPH_OPEN, kernel_thin)

    # --- 3. 自适应掩码 (保持之前的动态逻辑) ---
    valid_depths = depth_f[depth_raw > 0]
    foreground_mask = np.zeros((h, w), dtype=np.uint8)
    if len(valid_depths) > 0:
        d_min = np.percentile(valid_depths, 1)
        d_max = np.percentile(valid_depths, 99)
        dynamic_span = (d_max - d_min) * 0.6
        far_p = d_min + max(600, dynamic_span)
        binary_mask = ((depth_f >= d_min) & (depth_f <= far_p)).astype(np.uint8) * 255
        # 适度膨胀以包含边缘
        binary_mask = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 200: foreground_mask[labels == i] = 255

    # 4. 引导融合
    mts_grad_refined = cv2.bitwise_and(mts_grad_refined_raw, mts_grad_refined_raw, mask=foreground_mask)
    
    depth_display = cv2.normalize(depth_f, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_grad = get_sharp_grad(depth_display).astype(np.uint8)
    
    # 提取内部几何边缘
    diff = cv2.subtract(mts_grad_refined, depth_grad)
    # 使用大津法自适应提取细线条
    _, internal_ridges = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. 可视化对比
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1); plt.title("Original MTS (With RGB Ghosting)"); plt.imshow(cv2.cvtColor(mts_img, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 2); plt.title("Refined Sharp Edges (Anti-Ghosting)"); plt.imshow(mts_grad_refined, cmap='gray')
    
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 1] = depth_grad  # 绿色：深度跳变
    overlay[:, :, 2] = internal_ridges # 红色：MTS内部棱线
    
    plt.subplot(2, 3, 3); plt.title("Internal Ridge Detection"); plt.imshow(overlay)
    
    # 局部放大图预览（为了看清去重影效果）
    roi_y, roi_x = h//4, w//4
    plt.subplot(2, 3, 4); plt.title("Zoomed Original"); plt.imshow(cv2.cvtColor(mts_img[roi_y:roi_y+150, roi_x:roi_x+150], cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 5); plt.title("Zoomed Sharp Edge"); plt.imshow(mts_grad_refined[roi_y:roi_y+150, roi_x:roi_x+150], cmap='hot')
    plt.subplot(2, 3, 6); plt.title("Final Representation"); plt.imshow(cv2.addWeighted(depth_display, 0.4, mts_grad_refined, 0.9, 0), cmap='magma')

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()

# 执行
process_guided_6d_edges_sharp("../ycb_ev_data/dataset/test_pbr/000000/rgb_events/000003.png", 
                              "../ycb_ev_data/dataset/test_pbr/000000/depth/000003.png")