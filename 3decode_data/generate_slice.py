import zstandard as zstd
import numpy as np
import os
import cv2
import glob
from tqdm import tqdm

# ================= 配置 =================
RAW_DIR = "../ycb_ev_data/dataset/test_pbr/000000/ev_raw"
OUTPUT_DIR = "../ycb_ev_data/dataset/test_pbr/000000/rgb_events"

# 【关键修改】SD 分辨率通常是 VGA
WIDTH, HEIGHT = 640, 480 
# =======================================

def read_and_decode(path):
    with open(path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            data = reader.read()
    
    all_data = np.frombuffer(data, dtype=np.int32)
    
    # 智能跳过 Header
    # 如果前两个数看起来像分辨率 (比如 640, 480 或者 346, 260)，跳过
    if all_data.size > 2 and all_data[0] < 5000 and all_data[1] < 5000:
        # print(f"Found Header: {all_data[0]}x{all_data[1]}") # 调试用
        all_data = all_data[2:]
        
    valid_len = (all_data.size // 2) * 2
    events_raw = all_data[:valid_len].reshape(-1, 2)
    
    t = events_raw[:, 0].astype(float)
    packed_data = events_raw[:, 1]
    
    # 解码
    x = packed_data & 0x3FFF
    y = (packed_data >> 14) & 0x3FFF
    
    return x, y, t

def generate_rgb_stack(x, y, t, width, height):
    # 【新增】动态画布检查
    # 如果数据里的坐标比预设的 WIDTH/HEIGHT 还大，自动扩大画布
    # 这样你就永远不会看到“只有 1/4”的情况了
    max_x, max_y = x.max(), y.max()
    if max_x >= width: width = max_x + 1
    if max_y >= height: height = max_y + 1
    
    # 过滤越界 (此时理论上不会有越界了)
    mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t = x[mask], y[mask], t[mask]
    
    if len(x) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    # 时间归一化
    if t.max() == t.min():
        t_norm = np.zeros_like(t)
    else:
        t_norm = (t - t.min()) / (t.max() - t.min())
    
    img = np.zeros((height, width, 3), dtype=np.float32)
    
    # R (Past) -> G -> B (Future)
    mask_r = t_norm < 0.33
    np.add.at(img[:, :, 2], (y[mask_r], x[mask_r]), 1) 
    
    mask_g = (t_norm >= 0.33) & (t_norm < 0.66)
    np.add.at(img[:, :, 1], (y[mask_g], x[mask_g]), 1)
    
    mask_b = t_norm >= 0.66
    np.add.at(img[:, :, 0], (y[mask_b], x[mask_b]), 1)
    
    # 可视化增强
    img = np.log1p(img)
    if img.max() > 0:
        img = img / img.max() * 255
    
    return img.astype(np.uint8)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    zst_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.zst")))
    print(f"Processing {len(zst_files)} files...")
    
    # 先跑一张看看尺寸
    if len(zst_files) > 0:
        sample_x, sample_y, _ = read_and_decode(zst_files[0])
        print(f"\n【尺寸诊断】 数据中最大 X: {sample_x.max()}, 最大 Y: {sample_y.max()}")
        print(f"【尺寸诊断】 预设画布: {WIDTH} x {HEIGHT}")
        if sample_x.max() > WIDTH or sample_y.max() > HEIGHT:
            print("⚠️ 警告：数据尺寸超过预设！脚本将自动调整画布大小。")
        else:
            print("✅ 尺寸正常。")

    for fpath in tqdm(zst_files): 
        try:
            x, y, t = read_and_decode(fpath)
            rgb_img = generate_rgb_stack(x, y, t, WIDTH, HEIGHT)
            
            fname = os.path.basename(fpath).split('.')[0] + ".png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, fname), rgb_img)
            
        except Exception as e:
            print(f"Error {fpath}: {e}")

if __name__ == "__main__":
    main()