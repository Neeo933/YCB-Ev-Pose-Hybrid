import zstandard as zstd
import numpy as np
import os
import cv2
import glob
from tqdm import tqdm

# ================= 🏭 生产线配置 ================= 生成所有MTS图像
# 指向 dataset 的根目录 (包含 000000, 000001 等子文件夹的目录)
DATASET_ROOT = "../ycb_ev_data/dataset/test_pbr"

# 输出文件夹名字 (会自动在每个物体文件夹下创建这个目录)
OUTPUT_FOLDER_NAME = "rgb_events"

# 分辨率 (VGA Standard)
WIDTH, HEIGHT = 640, 480 
# ===============================================

def read_and_decode(path):
    """读取并解码 .zst 文件"""
    try:
        with open(path, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                data = reader.read()
        
        all_data = np.frombuffer(data, dtype=np.int32)
        
        # 智能 Header 跳过
        if all_data.size > 2 and all_data[0] < 5000 and all_data[1] < 5000:
            all_data = all_data[2:]
            
        valid_len = (all_data.size // 2) * 2
        events_raw = all_data[:valid_len].reshape(-1, 2)
        
        t = events_raw[:, 0].astype(float)
        packed_data = events_raw[:, 1]
        
        # 位运算解码 (BOP Standard)
        x = packed_data & 0x3FFF
        y = (packed_data >> 14) & 0x3FFF
        
        return x, y, t
    except Exception as e:
        # print(f"读取损坏: {path}")
        return None, None, None

def generate_rgb_stack(x, y, t, width, height):
    """生成 RGB 时序切片图"""
    # 动态画布调整 (防止越界)
    max_x, max_y = x.max(), y.max()
    if max_x >= width: width = max_x + 1
    if max_y >= height: height = max_y + 1
    
    mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t = x[mask], y[mask], t[mask]
    
    if len(x) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

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
    
    img = np.log1p(img)
    if img.max() > 0:
        img = img / img.max() * 255
    
    return img.astype(np.uint8)

def process_object_folder(obj_path):
    """处理单个物体文件夹"""
    raw_dir = os.path.join(obj_path, "ev_raw")
    out_dir = os.path.join(obj_path, OUTPUT_FOLDER_NAME)
    
    if not os.path.exists(raw_dir):
        return 0
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    zst_files = sorted(glob.glob(os.path.join(raw_dir, "*.zst")))
    count = 0
    
    # 不显示内部循环的进度条，避免刷屏，只在出错时打印
    for fpath in zst_files:
        # 构造输出文件名
        fname = os.path.basename(fpath).split('.')[0] + ".png"
        out_path = os.path.join(out_dir, fname)
        
        # 【断点续传】如果文件已存在且大小正常，跳过
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue
            
        x, y, t = read_and_decode(fpath)
        if x is not None:
            rgb_img = generate_rgb_stack(x, y, t, WIDTH, HEIGHT)
            cv2.imwrite(out_path, rgb_img)
            count += 1
            
    return count

def main():
    print(f"🚀 启动大规模数据生产线...")
    print(f"源目录: {DATASET_ROOT}")
    
    # 获取所有物体文件夹 (000000, 000001, ...)
    # 只处理数字命名的文件夹
    obj_ids = sorted([d for d in os.listdir(DATASET_ROOT) 
                      if os.path.isdir(os.path.join(DATASET_ROOT, d)) and d.isdigit()])
    
    print(f"发现 {len(obj_ids)} 个物体序列。")
    
    total_generated = 0
    
    # 主进度条
    pbar = tqdm(obj_ids, desc="Processing Objects")
    for obj_id in pbar:
        obj_path = os.path.join(DATASET_ROOT, obj_id)
        
        # 更新进度条描述
        pbar.set_description(f"Processing {obj_id}")
        
        # 处理该物体
        num = process_object_folder(obj_path)
        total_generated += num
        
    print(f"\n✅ 所有任务完成！")
    print(f"共生成 {total_generated} 张 RGB 时空切片图。")
    print(f"数据已就绪，准备开始训练！")

if __name__ == "__main__":
    main()