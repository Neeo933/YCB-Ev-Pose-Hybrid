import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
from pathlib import Path

class GMGPoseDataset(Dataset):
    def __init__(self, processed_dir, dataset_root, target_size=(128, 128), mode='train'):
        self.processed_dir = Path(processed_dir)
        self.dataset_root = Path(dataset_root)
        self.target_size = target_size
        self.mode = mode
        
        # 预先扫描所有样本
        self.sample_list = self._build_sample_list()
        print(f"Dataset loaded: {len(self.sample_list)} samples found in {mode} mode.")

    def _build_sample_list(self):
        samples = []
        label_root = self.processed_dir / "labels"
        
        # 统计一共跳过了多少坏样本
        skip_count = 0 
        
        for scene_dir in label_root.iterdir():
            if not scene_dir.is_dir(): continue
            scene_id = scene_dir.name
            
            # 加载元数据
            gt_path = self.dataset_root / scene_id / "scene_gt.json"
            info_path = self.dataset_root / scene_id / "scene_gt_info.json"
            
            with open(gt_path, 'r') as f: gt_data = json.load(f)
            with open(info_path, 'r') as f: info_data = json.load(f)
                
            for npy_path in scene_dir.glob("*.npy"):
                parts = npy_path.stem.split('_')
                frame_id_int = int(parts[0].replace('frame', ''))
                ins_idx = int(parts[1].replace('ins', ''))
                
                str_key = str(frame_id_int)
                if str_key not in gt_data: continue

                # 核心过滤逻辑
                info_item = info_data[str_key][ins_idx]
                bbox = info_item["bbox_visib"] 
                visib_fract = info_item.get("visib_fract", 0.0)

                if visib_fract < 0.5: 
                    skip_count += 1
                    continue

                # # 过滤极小的物体 (例如像素面积小于 1000)
                # # 防止 crop 出来全是马赛克
                # if bbox[2] * bbox[3] < 1000:
                #     skip_count += 1
                #     continue
                    
                if bbox[2] <= 0 or bbox[3] <= 0:
                    skip_count += 1
                    continue

                obj_id = gt_data[str_key][ins_idx]["obj_id"]
                
                samples.append({
                    'scene_id': scene_id,
                    'frame_id': frame_id_int,
                    'obj_id': obj_id,
                    'npy_path': npy_path,
                    'rgb_path': self.dataset_root / scene_id / "rgb" / f"{frame_id_int:06d}.jpg",
                    # [新增] 深度图路径 (YCB PBR 数据通常在 depth 文件夹下，png格式)
                    'depth_path': self.dataset_root / scene_id / "depth" / f"{frame_id_int:06d}.png",
                    'mts_path': self.dataset_root / scene_id / "rgb_events" / f"{frame_id_int:06d}.png",
                    'bbox': bbox 
                })
        
        print(f"过滤掉了 {skip_count} 个无效/低可见度样本。")
        return samples

    def _pad_bbox(self, bbox, img_w, img_h, padding_ratio=0.15):
        """给 BBox 增加 Padding，避免边缘效应"""
        x, y, w, h = bbox
        pad_w = int(w * padding_ratio)
        pad_h = int(h * padding_ratio)
        
        x_new = max(0, x - pad_w)
        y_new = max(0, y - pad_h)
        w_new = min(img_w - x_new, w + 2 * pad_w)
        h_new = min(img_h - y_new, h + 2 * pad_h)
        
        return [x_new, y_new, w_new, h_new]

    def _generate_vector_field(self, kpts_local):
        """生成向量场"""
        w, h = self.target_size
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        field = np.zeros((h, w, 18), dtype=np.float32)
        
        for i in range(9):
            target_x, target_y = kpts_local[i]
            dx = target_x - grid_x
            dy = target_y - grid_y
            mag = np.sqrt(dx**2 + dy**2) + 1e-6 
            field[:, :, i*2] = dx / mag
            field[:, :, i*2+1] = dy / mag
            
        return field.transpose(2, 0, 1)
    
    def __len__(self):
        return len(self.sample_list)

   
    def __getitem__(self, idx):
        s = self.sample_list[idx]
        
        # 1. 读取 RGB, MTS, Depth
        rgb = cv2.imread(str(s['rgb_path']))
        mts = cv2.imread(str(s['mts_path']), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(s['depth_path']), cv2.IMREAD_UNCHANGED)
        
        # 容错处理
        img_h, img_w = rgb.shape[:2]
        if mts is None: mts = np.zeros((img_h, img_w), dtype=np.uint8)
        if depth is None: depth = np.zeros((img_h, img_w), dtype=np.uint16)
        
        # 2. BBox 处理 (先获取原始 BBox 用于生成 Mask)
        bx, by, bw, bh = s['bbox']
        
        # === [核心改进] 使用 Otsu 算法生成精细 Mask ===
        # 步骤 A: 安全裁剪深度图
        bx_safe, by_safe = max(0, bx), max(0, by)
        bw_safe = min(bw, img_w - bx_safe)
        bh_safe = min(bh, img_h - by_safe)
        
        depth_crop_raw = depth[by_safe:by_safe+bh_safe, bx_safe:bx_safe+bw_safe]
        
        # 步骤 B: Otsu 自动分割
        mask_crop = np.zeros_like(depth_crop_raw, dtype=np.uint8)
        
        if depth_crop_raw.size > 0:
            # 1. 提取有效深度值 (非0)
            valid_mask = depth_crop_raw > 0
            valid_pixels = depth_crop_raw[valid_mask]
            
            if valid_pixels.size > 0:
                # 2. 归一化到 0-255 以便使用 OpenCV 的 Otsu
                d_min = valid_pixels.min()
                d_max = valid_pixels.max()
                
                if d_max - d_min > 5: # 如果深度有差异才分割，否则全是物体
                    # 线性映射: (d - min) / (max - min) * 255
                    # 注意要转成 float 计算再转 uint8
                    norm_depth = (depth_crop_raw.astype(np.float32) - d_min) / (d_max - d_min + 1e-6) * 255
                    norm_depth = norm_depth.astype(np.uint8)
                    
                    # 3. Otsu 二值化
                    # THRESH_BINARY_INV: 我们要的是“近”的物体 (值小 -> 颜色暗)
                    # Otsu 会把“暗”的部分(物体)和“亮”的部分(背景)分开
                    # 所以用 INV: 小于阈值(近)的设为 255，大于阈值(远)的设为 0
                    # 此外，原始深度为0的地方在 norm_depth 里也是 0 (最暗)，会被误判为物体
                    # 所以我们需要先对 valid 区域做处理，或者最后用 valid_mask 过滤
                    
                    # 为了稳健，我们只对 valid 的像素做 Otsu
                    # 但为了简单，我们直接做，然后把 0 值剔除
                    thresh_val, otsu_mask = cv2.threshold(
                        norm_depth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )
                    
                    # 4. 修正：原始深度为 0 的地方不是物体
                    mask_crop = otsu_mask
                    mask_crop[~valid_mask] = 0
                    
                else:
                    # 深度差异太小，说明全是物体 (或者全是背景)
                    mask_crop[valid_mask] = 255
        
        # 步骤 C: 把 crop 的 mask 放回全图
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        full_mask[by_safe:by_safe+bh_safe, bx_safe:bx_safe+bw_safe] = mask_crop
        # ===============================================
    
        # 3. 统一 Padding 和 最终裁剪 (使用 pad 后的框)
        x, y, w, h = self._pad_bbox(s['bbox'], img_w, img_h)
        
        crop_rgb = rgb[y:y+h, x:x+w]
        crop_mts = mts[y:y+h, x:x+w]
        crop_depth = depth[y:y+h, x:x+w]
        crop_mask = full_mask[y:y+h, x:x+w] 
        
        # 4. Resize (保持 Nearest 插值)
        crop_rgb = cv2.resize(crop_rgb, self.target_size)
        crop_mts = cv2.resize(crop_mts, self.target_size)
        crop_depth = cv2.resize(crop_depth, self.target_size, interpolation=cv2.INTER_NEAREST)
        crop_mask = cv2.resize(crop_mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # 5. 生成向量场
        kpts_orig = np.load(s['npy_path'])
        kpts_local = kpts_orig.copy()
        scale_x = self.target_size[0] / max(w, 1)
        scale_y = self.target_size[1] / max(h, 1)
        kpts_local[:, 0] = (kpts_orig[:, 0] - x) * scale_x
        kpts_local[:, 1] = (kpts_orig[:, 1] - y) * scale_y
        
        vector_field = self._generate_vector_field(kpts_local)
        
        # 6. Tensor化
        input_tensor = np.concatenate([
            crop_rgb.transpose(2, 0, 1) / 255.0, 
            crop_mts[np.newaxis, ...] / 255.0
        ], axis=0) 
        
        # 深度单位转换 0.1mm -> m
        depth_tensor = crop_depth.astype(np.float32) / 10000.0
        depth_tensor = depth_tensor[np.newaxis, ...] 
        
        # Mask
        mask_tensor = (crop_mask > 128).astype(np.float32)
        mask_tensor = mask_tensor[np.newaxis, ...]
    
        return {
            'input': torch.as_tensor(input_tensor, dtype=torch.float32),
            'depth': torch.as_tensor(depth_tensor, dtype=torch.float32),
            'target_field': torch.as_tensor(vector_field, dtype=torch.float32),
            'mask': torch.as_tensor(mask_tensor, dtype=torch.float32),
            'obj_id': torch.tensor(s['obj_id'], dtype=torch.long),
            'kpts_local': torch.as_tensor(kpts_local, dtype=torch.float32),
            'bbox': torch.tensor(s['bbox'], dtype=torch.float32), # 用于还原坐标
            'rgb_path': str(s['rgb_path']) # 用于读取原图可视化，转为str防止路径对象报错
        }