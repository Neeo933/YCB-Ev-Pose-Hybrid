import zstandard as zstd
import numpy as np
import os
import cv2
import glob
from tqdm import tqdm

# ================= 配置 =================
RAW_DIR = "../ycb_ev_data/dataset/test_pbr/000000/ev_raw"
OUTPUT_DIR = "../ycb_ev_data/dataset/test_pbr/000000/rgb_events"
WIDTH, HEIGHT = 346, 260
# =======================================

def read_and_decode(path):
    with open(path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            data = reader.read()
    
    # 1. 读取 int32
    all_data = np.frombuffer(data, dtype=np.int32)
    
    # 2. 智能 Header 处理
    # Col 0 是时间，Col 1 是数据。时间通常从小开始 (比如 71, 73)
    # 如果前两个数看起来像 [Width, Height] (比如 346, 260)，则跳过
    if all_data.size > 2 and all_data[0] == 346 and all_data[1] == 260:
        all_data = all_data[2:]
        
    # 3. Reshape
    # 确保偶数长度
    valid_len = (all_data.size // 2) * 2
    events_raw = all_data[:valid_len].reshape(-1, 2)
    
    # 4. 正确的列分配 (Col 0=Time, Col 1=Data)
    t = events_raw[:, 0].astype(float)
    packed_data = events_raw[:, 1]
    
    # 5. 正确的位运算解码 (BOP/YCB-Ev 标准)
    # 格式: (p << 28) | (y << 14) | x
    # 0x3FFF 是 14位掩码 (16383)
    x = packed_data & 0x3FFF
    y = (packed_data >> 14) & 0x3FFF
    # p = (packed_data >> 28) & 1 # 极性，暂时不用
    
    return x, y, t

def generate_rgb_stack(x, y, t, width, height):
    # 过滤越界
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