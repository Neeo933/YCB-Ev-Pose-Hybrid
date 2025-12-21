import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
import random
import glob

class GMGDataVerifier:
    def __init__(self, data_root):
        self.root = Path(data_root)
        self.templates_dir = self.root / "templates"
        self.labels_dir = self.root / "labels"
        
        # 你的代码中定义的关键点连接关系，用于画出立方体骨架验证几何关系
        # 0: (+,+,+), 1: (+,+,-), 2: (+,-,+), 3: (+,-,-)
        # 4: (-,+,+), 5: (-,+,-), 6: (-,-,+), 7: (-,-,-)
        self.edges = [
            (0,1), (2,3), (4,5), (6,7), # Z-axis lines
            (0,2), (1,3), (4,6), (5,7), # Y-axis lines
            (0,4), (1,5), (2,6), (3,7)  # X-axis lines
        ]

    def check_structure(self):
        print("=== 1. 检查目录结构 ===")
        if not self.root.exists():
            print(f"[错误] 数据根目录不存在: {self.root}")
            return False
        if not self.templates_dir.exists():
            print(f"[警告] 模板目录不存在 (可能是因为没有物体的 visib > 0.8): {self.templates_dir}")
        else:
            print(f"[通过] 模板目录存在: {self.templates_dir}")
            
        if not self.labels_dir.exists():
            print(f"[错误] 标签目录不存在: {self.labels_dir}")
            return False
        else:
            print(f"[通过] 标签目录存在: {self.labels_dir}")
        return True

    def check_templates(self, sample_ratio=0.1):
        print("\n=== 2. 检查模板图像 (抽样检查) ===")
        img_files = list(self.templates_dir.rglob("*.png"))
        if not img_files:
            print("[警告] 没有找到任何模板图片 (.png)")
            return

        print(f"共发现 {len(img_files)} 张模板图片，将抽查 {int(len(img_files)*sample_ratio) + 1} 张...")
        
        valid_cnt = 0
        samples = random.sample(img_files, max(1, int(len(img_files) * sample_ratio)))
        
        for img_path in samples:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[失败] 无法读取图片: {img_path}")
                continue
            
            if img.shape[0] != 128 or img.shape[1] != 128:
                print(f"[失败] 图片尺寸错误 {img.shape} (期望 128x128): {img_path}")
                continue
                
            if img.mean() < 1.0: # 简单的检查，看是不是全黑
                print(f"[警告] 图片几乎全黑 (mean < 1): {img_path}")
            
            valid_cnt += 1
            
        print(f"[结果] {valid_cnt}/{len(samples)} 张图片检查通过。")

    def check_labels(self):
        print("\n=== 3. 检查标签数据 (.npy) ===")
        npy_files = list(self.labels_dir.rglob("*.npy"))
        if not npy_files:
            print("[错误] 没有找到任何标签文件 (.npy)")
            return

        print(f"共发现 {len(npy_files)} 个标签文件。")
        
        invalid_files = []
        for npy_path in npy_files:
            try:
                kpts = np.load(npy_path)
                
                # 检查形状
                if kpts.shape != (9, 2):
                    print(f"[形状错误] {npy_path.name}: {kpts.shape} != (9, 2)")
                    invalid_files.append(npy_path)
                    continue
                
                # 检查 NaN 或 Inf
                if not np.isfinite(kpts).all():
                    print(f"[数值错误] {npy_path.name} 包含 NaN 或 Inf")
                    invalid_files.append(npy_path)
                    continue
                    
            except Exception as e:
                print(f"[读取错误] {npy_path}: {e}")
                invalid_files.append(npy_path)

        if len(invalid_files) == 0:
            print(f"[通过] 所有 {len(npy_files)} 个标签文件格式正确。")
        else:
            print(f"[失败] 发现 {len(invalid_files)} 个有问题的文件。")

    def visualize_random_sample(self):
        print("\n=== 4. 可视化验证 (几何一致性) ===")
        # 随机选取一个标签文件
        npy_files = list(self.labels_dir.rglob("*.npy"))
        if not npy_files: return

        sample_npy = random.choice(npy_files)
        kpts = np.load(sample_npy)
        
        print(f"正在可视化样本: {sample_npy}")
        print(f"数据范围: X[{kpts[:,0].min():.1f}, {kpts[:,0].max():.1f}], Y[{kpts[:,1].min():.1f}, {kpts[:,1].max():.1f}]")
        
        # 创建画布 (因为我们可能没有原始全图，就创建一个足够大的空白画布来画点)
        # 假设原始分辨率是 VGA (640x480) 或 HD，根据点的范围自适应
        canvas_w = int(max(640, kpts[:, 0].max() + 50))
        canvas_h = int(max(480, kpts[:, 1].max() + 50))
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) + 255 # 白色背景

        # 画关键点
        for i, (x, y) in enumerate(kpts):
            color = (0, 0, 255) if i < 8 else (255, 0, 0) # 0-7 红色 (角点), 8 蓝色 (中心)
            cv2.circle(canvas, (int(x), int(y)), 3, color, -1)
            cv2.putText(canvas, str(i), (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

        # 画连接线 (验证是否像个立方体)
        for start_idx, end_idx in self.edges:
            pt1 = tuple(kpts[start_idx].astype(int))
            pt2 = tuple(kpts[end_idx].astype(int))
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 2) # 绿色连线

        # 显示
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.title(f"Check: {sample_npy.name}\nDo you see a 3D box structure?")
        plt.axis('on')
        plt.show()

if __name__ == "__main__":
    # 指向你的输出目录
    output_dir = "./processed_data" 
    
    verifier = GMGDataVerifier(output_dir)
    
    if verifier.check_structure():
        verifier.check_templates()
        verifier.check_labels()
        verifier.visualize_random_sample()