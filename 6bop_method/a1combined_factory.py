import numpy as np
import cv2
import json
import os
from pathlib import Path
from tqdm import tqdm

class GMGDataFactory:
    def __init__(self, dataset_root, output_dir="processed_data"):
        self.root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.templates_dir = self.output_dir / "templates"
        self.labels_dir = self.output_dir / "labels"
        
        # 创建输出目录
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 虚拟参考物体尺寸 (用于定义 9 个伪关键点)
        self.ref_size = 50.0  # 50mm

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def get_virtual_pts_3d(self):
        """定义物体坐标系下的 9 个关键点"""
        s = self.ref_size
        return np.array([
            [s,s,s], [s,s,-s], [s,-s,s], [s,-s,-s],
            [-s,s,s], [-s,s,-s], [-s,-s,s], [-s,-s,-s],
            [0,0,0] # 质心
        ])

    def process_scene(self, scene_id):
        scene_path = self.root / scene_id
        if not scene_path.exists():
            print(f"[Error] 场景路径不存在: {scene_path}")
            return

        # 加载元数据
        cam_data = self._load_json(scene_path / "scene_camera.json")
        gt_data = self._load_json(scene_path / "scene_gt.json")
        info_data = self._load_json(scene_path / "scene_gt_info.json")

        # 遍历每一帧
        for img_id_str in tqdm(gt_data.keys(), desc=f"Scene {scene_id}", leave=False):
            img_idx = int(img_id_str)
            
            # 1. 加载图像资源
            rgb_path = scene_path / "rgb" / f"{img_idx:06d}.jpg"
            mts_path = scene_path / "rgb_events" / f"{img_idx:06d}.png"
            depth_path = scene_path / "depth" / f"{img_idx:06d}.png"

            rgb_img = cv2.imread(str(rgb_path))
            if rgb_img is None: continue
            
            mts_img = cv2.imread(str(mts_path))
            
            # 懒加载深度图：仅在这一帧有符合条件的物体时才读取
            depth_img = None 

            # 获取当前帧相机内参 K
            K = np.array(cam_data[img_id_str]["cam_K"]).reshape(3, 3)
            
            # 遍历该帧中的所有物体实例
            for i, obj_gt in enumerate(gt_data[img_id_str]):
                obj_id = obj_gt["obj_id"]
                obj_info = info_data[img_id_str][i]
                
                # --- A. 记录关键点标签 (Vector Field 训练用) ---
                R = np.array(obj_gt["cam_R_m2c"]).reshape(3, 3)
                t = np.array(obj_gt["cam_t_m2c"]).reshape(3, 1)
                pts_3d = self.get_virtual_pts_3d()
                pts_2d_homo = (K @ (R @ pts_3d.T + t)).T
                pts_2d = pts_2d_homo[:, :2] / pts_2d_homo[:, 2:]
                
                scene_label_dir = self.labels_dir / scene_id
                scene_label_dir.mkdir(parents=True, exist_ok=True)
                np.save(scene_label_dir / f"frame{img_idx}_ins{i}_kpts.npy", pts_2d)

                # --- B. 模板提取 (RGB, MTS, Depth) ---
                # 策略：如果物体可见度 > 80%，则截取模板
                if obj_info["visib_fract"] > 0.8:
                    x, y, w, h = obj_info["bbox_visib"]
                    obj_tpl_dir = self.templates_dir / f"obj_{obj_id}"
                    obj_tpl_dir.mkdir(parents=True, exist_ok=True)
                    
                    base_name = f"scene{scene_id}_frame{img_idx}_ins{i}"

                    # 1. 处理 RGB
                    crop_rgb = rgb_img[y:y+h, x:x+w]
                    if crop_rgb.size > 0:
                        cv2.imwrite(str(obj_tpl_dir / f"{base_name}_rgb.png"), 
                                    cv2.resize(crop_rgb, (128, 128)))

                    # 2. 处理 MTS (如果存在)
                    if mts_img is not None:
                        crop_mts = mts_img[y:y+h, x:x+w]
                        if crop_mts.size > 0:
                            cv2.imwrite(str(obj_tpl_dir / f"{base_name}_mts.png"), 
                                        cv2.resize(crop_mts, (128, 128)))

                    # 3. 处理 Depth (补全逻辑)
                    if depth_img is None and depth_path.exists():
                        # 读取原始 16-bit 深度图
                        depth_img = cv2.imread(str(depth_path), -1)
                    
                    if depth_img is not None:
                        crop_depth = depth_img[y:y+h, x:x+w]
                        if crop_depth.size > 0:
                            # 深度图缩放必须使用 INTER_NEAREST 保持数值准确
                            crop_depth_res = cv2.resize(crop_depth, (128, 128), 
                                                       interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(str(obj_tpl_dir / f"{base_name}_depth.png"), crop_depth_res)

        print(f"场景 {scene_id} 处理完成。")

if __name__ == "__main__":
    # 配置路径
    DATASET_ROOT = "../ycb_ev_data/dataset/test_pbr"
    OUTPUT_DIR = "./processed_data"
    
    # 初始化工厂
    factory = GMGDataFactory(dataset_root=DATASET_ROOT, output_dir=OUTPUT_DIR)
    
    # 定义要处理的场景
    scenes = ["000000", "000001", "000002", "000003", "000004"]
    
    print("--- 开始生成多模态数据与标签 ---")
    for s_id in tqdm(scenes, desc="总体进度"):
        try:
            factory.process_scene(s_id)
        except Exception as e:
            print(f"\n[Error] 处理场景 {s_id} 时发生异常: {e}")
            
    print("--- 任务全部完成！ ---")