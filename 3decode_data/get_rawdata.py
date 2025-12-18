import zstandard as zstd
import numpy as np
import os
import glob

# ================= 配置 =================
# 指向你的 ev_raw 文件夹
RAW_DIR = "../ycb_ev_data/dataset/test_pbr/000000/ev_raw"
# =======================================

def inspect_file(path):
    print(f"\n正在检查文件: {os.path.basename(path)}")
    
    with open(path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            data = reader.read()
            
    # 转为 int32 数组
    raw_ints = np.frombuffer(data, dtype=np.int32)
    
    print(f"1. 数据总长度 (int32个数): {raw_ints.size}")
    
    # 打印前 20 个原始整数
    print("\n2. 前 20 个原始整数 (Raw Integers):")
    for i, val in enumerate(raw_ints[:20]):
        print(f"   [{i}]: {val}")

    # ----------------------------------------------------
    # 假设测试 A: 存在 Header [Width, Height]
    # ----------------------------------------------------
    print("\n3. Header 假设测试:")
    w_guess = raw_ints[0]
    h_guess = raw_ints[1]
    print(f"   假设前两个是分辨率: W={w_guess}, H={h_guess}")
    
    # ----------------------------------------------------
    # 假设测试 B: 数据是 2 列结构 [Packed, Time]
    # ----------------------------------------------------
    # 如果有 header，我们跳过前 2 个；如果没有，就不跳
    # 我们看后续的数据
    start_idx = 2 if (w_guess < 10000 and h_guess < 10000) else 0
    
    print(f"\n4. 尝试按 (N, 2) 打印前 10 行 (跳过Header={start_idx}):")
    try:
        data_body = raw_ints[start_idx:]
        # 强制变成偶数长度以便 reshape
        if data_body.size % 2 != 0:
            data_body = data_body[:-1]
        
        reshaped = data_body.reshape(-1, 2)
        
        print(f"   {'索引':<5} | {'Col 0 (Packed?)':<20} | {'Col 1 (Time?)':<20}")
        print("   " + "-"*50)
        for i in range(10):
            v0 = reshaped[i, 0]
            v1 = reshaped[i, 1]
            print(f"   {i:<5} | {v0:<20} | {v1:<20}")

        # ----------------------------------------------------
        # 假设测试 C: 尝试用 Width=346 解码 Col 0
        # ----------------------------------------------------
        print("\n5. 尝试解码第一行数据 (假设 Width=346):")
        test_val = reshaped[0, 0]
        
        # 假设 1: 位运算
        x_bit = test_val & 511
        y_bit = test_val >> 9
        
        # 假设 2: 除法 (Flat Index)
        y_div = test_val // 346
        x_div = test_val % 346
        
        print(f"   原始值: {test_val}")
        print(f"   [假设1 位运算]: x={x_bit}, y={y_bit}")
        print(f"   [假设2 FlatIdx]: x={x_div}, y={y_div}")

    except Exception as e:
        print(f"   无法 Reshape: {e}")

def main():
    zst_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.zst")))
    if not zst_files:
        print("错误：未找到文件！")
        return
    
    # 只检查第一个文件
    inspect_file(zst_files[0])

if __name__ == "__main__":
    main()