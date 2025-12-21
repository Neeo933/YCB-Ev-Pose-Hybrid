import numpy as np
import cv2
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import random

class GMGVerifierWithTemplate:
    def __init__(self, dataset_root, processed_dir):
        self.dataset_root = Path(dataset_root)
        self.processed_dir = Path(processed_dir)
        self.labels_dir = self.processed_dir / "labels"
        self.templates_root = self.processed_dir / "templates"
        
        # 3D 框连接关系
        self.edges = [
            (0,1), (2,3), (4,5), (6,7), 
            (0,2), (1,3), (4,6), (5,7), 
            (0,4), (1,5), (2,6), (3,7) 
        ]

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def get_random_template(self, obj_id):
        """尝试获取该物体 ID 的任意一张模板图"""
        obj_tpl_dir = self.templates_root / f"obj_{obj_id}"
        
        # 创建一个空白/提示图片作为默认值
        placeholder = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Template", (10, 64), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if not obj_tpl_dir.exists():
            return placeholder, "Dir Not Found"
            
        tpl_files = list(obj_tpl_dir.glob("*.png"))
        # 过滤掉 mask 或 mts (如果有的话)，只看 rgb
        rgb_files = [f for f in tpl_files if "rgb" in f.name]
        
        if not rgb_files:
            return placeholder, "No Images"
            
        # 随机选一张
        chosen_file = random.choice(rgb_files)
        img = cv2.imread(str(chosen_file))
        if img is None:
            return placeholder, "Read Error"
            
        return img, chosen_file.name

    def draw_3d_box(self, img, kpts, obj_id):
        vis_img = img.copy()
        # 画点和线
        for idx, (x, y) in enumerate(kpts[:8]):
            cv2.circle(vis_img, (int(x), int(y)), 4, (0, 0, 255), -1)
        cx, cy = kpts[8]
        cv2.circle(vis_img, (int(cx), int(cy)), 5, (255, 0, 0), -1)
        for i, j in self.edges:
            pt1, pt2 = tuple(kpts[i].astype(int)), tuple(kpts[j].astype(int))
            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)
            
        # 在物体中心写个简短的 ID
        cv2.putText(vis_img, f"ID:{obj_id}", (int(cx), int(cy)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return vis_img

    def combine_views(self, scene_img, template_img, obj_id, tpl_name):
        """将场景图和模板图拼接到一起"""
        h, w, _ = scene_img.shape
        
        # 定义侧边栏宽度 (例如 256 像素)
        sidebar_w = 256
        sidebar = np.ones((h, sidebar_w, 3), dtype=np.uint8) * 50 # 深灰色背景
        
        # 1. 调整模板大小 (原始是 128x128，放大显示看得更清)
        tpl_disp_size = 200
        if template_img is not None:
            tpl_resized = cv2.resize(template_img, (tpl_disp_size, tpl_disp_size))
        else:
            tpl_resized = np.zeros((tpl_disp_size, tpl_disp_size, 3), dtype=np.uint8)
            
        # 2. 将模板贴到侧边栏中间
        y_offset = (h - tpl_disp_size) // 2
        x_offset = (sidebar_w - tpl_disp_size) // 2
        sidebar[y_offset:y_offset+tpl_disp_size, x_offset:x_offset+tpl_disp_size] = tpl_resized
        
        # 3. 添加文字说明
        cv2.putText(sidebar, f"Target Object", (10, y_offset - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(sidebar, f"ID: {obj_id}", (10, y_offset - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # 显示模板文件名（太长就截断）
        short_name = tpl_name[:20] + "..." if len(tpl_name) > 20 else tpl_name
        cv2.putText(sidebar, f"Ref: {short_name}", (10, y_offset + tpl_disp_size + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # 4. 拼接
        combined = np.hstack((scene_img, sidebar))
        return combined

    def verify_random_sample(self):
        # ... (前面的寻找逻辑和之前一样) ...
        scene_dirs = [d for d in self.labels_dir.iterdir() if d.is_dir()]
        if not scene_dirs: return
        scene_path = random.choice(scene_dirs)
        scene_id = scene_path.name
        
        gt_path = self.dataset_root / scene_id / "scene_gt.json"
        if not gt_path.exists(): return
        scene_gt = self._load_json(gt_path)
        
        npy_files = list(scene_path.glob("*.npy"))
        if not npy_files: return
        npy_file = random.choice(npy_files)
        
        # 解析 ID
        parts = npy_file.name.split('_')
        frame_idx = int(parts[0].replace('frame', ''))
        ins_idx = int(parts[1].replace('ins', ''))
        
        gt_key = str(frame_idx)
        if gt_key not in scene_gt: return
        obj_id = scene_gt[gt_key][ins_idx]["obj_id"]

        # 读取图片和关键点
        img_path = self.dataset_root / scene_id / "rgb" / f"{frame_idx:06d}.jpg"
        scene_img = cv2.imread(str(img_path))
        kpts = np.load(npy_file)
        
        if scene_img is None: return

        # === 核心逻辑 ===
        # 1. 在原图画框
        vis_scene = self.draw_3d_box(scene_img, kpts, obj_id)
        
        # 2. 获取该物体 ID 的任意一个模板
        tpl_img, tpl_name = self.get_random_template(obj_id)
        
        # 3. 拼图
        final_vis = self.combine_views(vis_scene, tpl_img, obj_id, tpl_name)

        # 4. 显示
        plt.figure(figsize=(14, 8))
        plt.imshow(cv2.cvtColor(final_vis, cv2.COLOR_BGR2RGB))
        plt.title(f"Scene: {scene_id} | Frame: {frame_idx} | Instance: {ins_idx}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # 配置你的路径
    DATASET_ROOT = "../ycb_ev_data/dataset/test_pbr"
    PROCESSED_DIR = "./processed_data"

    verifier = GMGVerifierWithTemplate(DATASET_ROOT, PROCESSED_DIR)
    
    print("正在抽取样本进行验证...")
    verifier.verify_random_sample()
    verifier.verify_random_sample()