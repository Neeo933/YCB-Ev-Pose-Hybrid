import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_three_channel_tensor(mts_path, depth_path, save_name="guided_6d_3channel_fusion.png"):
    # 1. 数据加载与基础处理
    mts_img = cv2.imread(mts_path)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if mts_img is None or depth_raw is None:
        print("路径错误，请检查文件。")
        return

    h, w = depth_raw.shape
    depth_f = depth_raw.astype(np.float32)

    # 2. MTS 通道对齐与去重影 (核心特征提取)
    b, g, r = cv2.split(mts_img)
    def get_grad(channel):
        gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    # 提取去重影后的锐利边缘
    grad_refined = get_grad(g) * 0.6 + get_grad(r) * 0.2 + get_grad(b) * 0.2
    grad_refined = cv2.normalize(grad_refined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. 自适应掩码 (背景剔除)
    valid_mask = depth_raw > 0
    foreground_mask = np.zeros((h, w), dtype=np.uint8)
    if np.any(valid_mask):
        d_min, d_max = np.percentile(depth_f[valid_mask], 1), np.percentile(depth_f[valid_mask], 99)
        far_p = d_min + max(0, (d_max - d_min) * 0.9)
        binary_mask = ((depth_f >= d_min) & (depth_f <= far_p)).astype(np.uint8) * 255
        binary_mask = cv2.dilate(binary_mask, np.ones((7, 7), np.uint8))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
        for i in range(1, num_labels):
            if 200 < stats[i, cv2.CC_STAT_AREA] < (h * w * 0.8):
                foreground_mask[labels == i] = 255

    # 4. 生成三个独立特征通道
    # 通道 1 (R): MTS 锐化边缘 (遮罩处理)
    channel_R = cv2.bitwise_and(grad_refined, grad_refined, mask=foreground_mask)
    
    # 通道 2 (G): 深度图梯度 (展现深度边缘缺失问题)
    depth_display = cv2.normalize(depth_f, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    channel_G = get_grad(depth_display).astype(np.uint8)
    
    # 通道 3 (B): 归一化深度图 (展示物体实体)
    channel_B = depth_display.copy()

    # --- 5. 【核心修改】合成三通道特征图 ---
    # 合成为一个 BGR 图像 (OpenCV默认顺序)
    three_channel_fusion = cv2.merge([channel_B, channel_G, channel_R])

    # 6. 可视化
    fig = plt.figure(figsize=(20, 10))
    
    # 子图1: 原始输入
    plt.subplot(1, 2, 1)
    plt.title("Raw Multi-Modal Inputs\n(MTS & Depth Scan)", fontweight='bold')
    plt.imshow(cv2.cvtColor(mts_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # 子图2: 三通道融合张量
    plt.subplot(1, 2, 2)
    plt.title("3-Channel Fusion Tensor\nRed: MTS Edges | Green: Depth Grad | Blue: Depth Value", fontweight='bold')
    # 注意：matplotlib 显示 RGB，需翻转通道
    plt.imshow(cv2.merge([channel_R, channel_G, channel_B])) 
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"三通道融合图像已保存: {save_name}")
    plt.show()

# 执行代码
D_PATH = "../ycb_ev_data/dataset/test_pbr/000000/depth/000005.png"
M_PATH = "../ycb_ev_data/dataset/test_pbr/000000/rgb_events/000005.png"
generate_three_channel_tensor(M_PATH, D_PATH)