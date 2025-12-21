import numpy as np
import cv2
import json
import os
from pathlib import Path
from tqdm import tqdm

class GMGDepthPatcher:
    def __init__(self, dataset_root, output_dir="data_factory_output"):
        self.root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.templates_dir = self.output_dir / "templates"
        
        # 检查输出目录是否存在
        if not self.templates_dir.exists():
            raise FileNotFoundError(f"找不到现有的模板目录: {self.templates_dir}，请先运行原来的 DataFactory。")

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def process_scene(self, scene_id):
        scene_path = self.root / scene_id
        
        # 1. 只需要加载 GT 和 Info
        # cam_data 不需要，labels 也不需要重新算
        gt_data = self._load_json(scene_path / "scene_gt.json")
        info_data = self._load_json(scene_path / "scene_gt_info.json")

        # 统计
        processed_count = 0
        skipped_count = 0

        # 遍历每一帧
        for img_id_str in tqdm(gt_data.keys(), desc=f"Scene {scene_id}", leave=False):
            img_idx = int(img_id_str)
            
            # 懒加载标记：只有当这一帧里确实需要补全深度时，才去读 Depth 大图
            depth_img = None 
            
            # 遍历该帧中的所有物体实例
            for i, obj_gt in enumerate(gt_data[img_id_str]):
                obj_id = obj_gt["obj_id"]
                
                # 构造 RGB 模板的路径 (用来判断之前是否生成过)
                # 路径格式: templates/obj_{id}/scene{id}_frame{idx}_ins{i}_rgb.png
                obj_tpl_dir = self.templates_dir / f"obj_{obj_id}"
                rgb_path = obj_tpl_dir / f"scene{scene_id}_frame{img_idx}_ins{i}_rgb.png"
                
                # 目标深度图路径
                depth_save_path = obj_tpl_dir / f"scene{scene_id}_frame{img_idx}_ins{i}_depth.png"

                # === 核心判断逻辑 ===
                # 1. 如果之前 RGB 没生成 (说明可见度低)，那 Depth 也不需要
                if not rgb_path.exists():
                    continue
                
                # 2. 如果 Depth 已经有了，跳过
                if depth_save_path.exists():
                    continue

                # === 开始补全 ===
                # 如果是这一帧第一次需要补全，加载 Depth 大图
                if depth_img is None:
                    # YCB PBR 的深度图通常在 depth 文件夹，格式是 16-bit PNG
                    depth_file = scene_path / "depth" / f"{img_idx:06d}.png"
                    if not depth_file.exists():
                        # 容错：有些数据集可能是 .jpg 或其他名字，视情况调整
                        print(f"[Warn] 深度图缺失: {depth_file}")
                        break
                    
                    # Flag -1 (IMREAD_UNCHANGED) 读取原始 16-bit 数据
                    depth_img = cv2.imread(str(depth_file), -1)
                    if depth_img is None: break

                # 获取 BBox 进行裁剪
                obj_info = info_data[img_id_str][i]
                x, y, w, h = obj_info["bbox_visib"]
                
                # 裁剪
                crop_depth = depth_img[y:y+h, x:x+w]
                
                if crop_depth.size > 0:
                    # Resize 到 128x128
                    # !!! 重要 !!! 深度图 Resize 必须用 INTER_NEAREST 或 INTER_AREA
                    # 绝对不能用默认的 Bilinear，否则会在边缘产生错误的深度值(鬼影)
                    crop_depth = cv2.resize(crop_depth, (128, 128), interpolation=cv2.INTER_NEAREST)
                    
                    # 保存
                    cv2.imwrite(str(depth_save_path), crop_depth)
                    processed_count += 1
                else:
                    skipped_count += 1

        print(f"场景 {scene_id}: 补全了 {processed_count} 张深度模板。")

if __name__ == "__main__":
    # 配置你的路径
    DATASET_ROOT = "../ycb_ev_data/dataset/test_pbr"
    PROCESSED_DIR = "./processed_data" # 你之前的输出目录
    
    patcher = GMGDepthPatcher(DATASET_ROOT, PROCESSED_DIR)
    
    # 定义场景
    scenes = ["000000", "000001", "000002", "000003", "000004"]
    
    print("--- 开始增量补全深度图 ---")
    for s_id in scenes:
        patcher.process_scene(s_id)
    print("--- 全部完成 ---")