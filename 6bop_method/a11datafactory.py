import numpy as np
import cv2
import json
import os
from pathlib import Path
from tqdm import tqdm

class GMGDataFactory:
    def __init__(self, dataset_root, output_dir="data_factory_output"):
        self.root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 虚拟参考物体尺寸 (用于定义 9 个伪关键点)
        self.ref_size = 50.0 # 50mm

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
        print(f"正在处理场景: {scene_id}...")

        # 加载元数据
        cam_data = self._load_json(scene_path / "scene_camera.json")
        gt_data = self._load_json(scene_path / "scene_gt.json")
        info_data = self._load_json(scene_path / "scene_gt_info.json")

        # 遍历每一帧
        for img_id_str in gt_data.keys():
            img_idx = int(img_id_str)
            
            # 获取当前帧相机内参 K 和深度缩放
            K = np.array(cam_data[img_id_str]["cam_K"]).reshape(3, 3)
            
            # 读取图像 (RGB + MTS)
            rgb_img = cv2.imread(str(scene_path / "rgb" / f"{img_idx:06d}.jpg"))
            # 假设你的 MTS 文件命名为 mts_000000.png
            mts_img = cv2.imread(str(scene_path / "rgb_events" / f"{img_idx:06d}.png"))
            
            if rgb_img is None: continue

            # 遍历该帧中的所有物体实例
            for i, obj_gt in enumerate(gt_data[img_id_str]):
                obj_id = obj_gt["obj_id"]
                obj_info = info_data[img_id_str][i]
                
                # 1. 计算伪关键点投影 (用于 Vector Field 训练)
                R = np.array(obj_gt["cam_R_m2c"]).reshape(3, 3)
                t = np.array(obj_gt["cam_t_m2c"]).reshape(3, 1)
                pts_3d = self.get_virtual_pts_3d()
                pts_2d_homo = (K @ (R @ pts_3d.T + t)).T
                pts_2d = pts_2d_homo[:, :2] / pts_2d_homo[:, 2:] # 关键点像素坐标

                # 2. 自动模板提取
                # 策略：如果物体可见度 > 80%，则截取作为 Model-Free 模板
                if obj_info["visib_fract"] > 0.8:
                    bbox = obj_info["bbox_visib"] # [x, y, w, h]
                    x, y, w, h = bbox
                    
                    # 裁剪并保存
                    crop_rgb = rgb_img[y:y+h, x:x+w]
                    crop_mts = mts_img[y:y+h, x:x+w] if mts_img is not None else None
                    
                    if crop_rgb.size > 0:
                        template_path = self.output_dir / "templates" / f"obj_{obj_id}"
                        template_path.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(template_path / f"scene{scene_id}_frame{img_idx}_ins{i}_rgb.png"), 
                                    cv2.resize(crop_rgb, (128, 128)))
                        if crop_mts is not None:
                            cv2.imwrite(str(template_path / f"scene{scene_id}_frame{img_idx}_ins{i}_mts.png"), 
                                        cv2.resize(crop_mts, (128, 128)))

                # 3. 记录关键点标签 (保存为 .npy 供 DataLoader 直接调用)
                label_path = self.output_dir / "labels" / scene_id
                label_path.mkdir(parents=True, exist_ok=True)
                np.save(label_path / f"frame{img_idx}_ins{i}_kpts.npy", pts_2d)

        print(f"场景 {scene_id} 处理完成。")

# 使用示例
# factory = GMGDataFactory(dataset_root="../ycb_ev_data/dataset/test_pbr")
# factory.process_scene("000000")

if __name__ == "__main__":
    # 初始化数据工厂
    factory = GMGDataFactory(dataset_root="../ycb_ev_data/dataset/test_pbr", output_dir="./processed_data")
    
    # 定义要处理的场景列表
    scenes = ["000000" ,"000001", "000002", "000003", "000004"]
    
    # 使用 tqdm 创建进度条
    for s_id in tqdm(scenes, desc="处理场景进度", unit="场景"):
        try:
            factory.process_scene(s_id)
        except Exception as e:
            print(f"处理场景 {s_id} 时出错: {e}")

    print("--- 所有 5 个场景预处理已完成！ ---")、