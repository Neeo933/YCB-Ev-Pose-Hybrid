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

                if visib_fract < 0.1: 
                    skip_count += 1
                    continue
                
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

    # def __getitem__(self, idx):
    #     s = self.sample_list[idx]
        
    #     # 1. 读取 RGB
    #     rgb = cv2.imread(str(s['rgb_path']))
    #     if rgb is None: raise ValueError(f"Image error: {s['rgb_path']}")
        
    #     # 2. 读取 MTS
    #     mts = cv2.imread(str(s['mts_path']), cv2.IMREAD_GRAYSCALE)
    #     if mts is None: mts = np.zeros_like(rgb[:,:,0])
        
    #     img_h, img_w = rgb.shape[:2]


    #     # === [新增/修改] 内存中实时生成 Mask ===
    #     # 创建一个全黑的底图
    #     mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
    #     # 获取原始 bbox
    #     bx, by, bw, bh = s['bbox']
        
    #     # 将 bbox 区域填白 (255)
    #     # 这就是告诉网络：在这个方框里的，都是需要预测向量的有效区域
    #     mask[by:by+bh, bx:bx+bw] = 255

    #     # 3. [新增] 读取 Depth (必须用 -1 读取原始 16-bit 数据)
    #     depth = cv2.imread(str(s['depth_path']), cv2.IMREAD_UNCHANGED)
    #     if depth is None: 
    #         # 容错：如果找不到深度图，给一个全黑的，避免报错
    #         depth = np.zeros_like(rgb[:,:,0], dtype=np.uint16)
        

    #     # 4. BBox Padding
    #     x, y, w, h = self._pad_bbox(s['bbox'], img_w, img_h)
        
    #     # 5. 裁剪 (Crop)
    #     crop_rgb = rgb[y:y+h, x:x+w]
    #     crop_mts = mts[y:y+h, x:x+w]
    #     crop_depth = depth[y:y+h, x:x+w] # [新增]
    #     crop_mask = mask[y:y+h, x:x+w]

    #     # 6. Resize 到网络输入尺寸 (128x128)
    #     crop_rgb = cv2.resize(crop_rgb, self.target_size)
    #     crop_mts = cv2.resize(crop_mts, self.target_size)
    #     crop_mask = cv2.resize(crop_mask, self.target_size, interpolation=cv2.INTER_NEAREST)

    #     # [新增] 深度图 Resize (注意：使用 INTER_NEAREST 保持物理意义)
    #     crop_depth = cv2.resize(crop_depth, self.target_size, interpolation=cv2.INTER_NEAREST)
        
    #     # 7. 关键点坐标变换
    #     kpts_orig = np.load(s['npy_path'])
    #     kpts_local = kpts_orig.copy()
        
    #     scale_x = self.target_size[0] / w
    #     scale_y = self.target_size[1] / h
    #     kpts_local[:, 0] = (kpts_orig[:, 0] - x) * scale_x
    #     kpts_local[:, 1] = (kpts_orig[:, 1] - y) * scale_y
        
    #     # 8. 生成向量场
    #     vector_field = self._generate_vector_field(kpts_local)
        
    #     # [新增] 处理 Mask Tensor
    #     # 转为 0/1 float, 形状 [1, 128, 128]
    #     mask_tensor = (crop_mask > 128).astype(np.float32)
    #     mask_tensor = mask_tensor[np.newaxis, ...]

    #     # 9. 整理 Tensor
    #     # Input: 4 Channels (RGB + MTS)
    #     input_tensor = np.concatenate([
    #         crop_rgb.transpose(2, 0, 1) / 255.0, 
    #         crop_mts[np.newaxis, ...] / 255.0
    #     ], axis=0) 
        
    #     # [新增] Depth Tensor 处理
    #     # 深度通常是 mm 单位，除以 1000 转为 米(m)，方便网络数值稳定
    #     depth_tensor = crop_depth.astype(np.float32) / 10000.0
    #     depth_tensor = depth_tensor[np.newaxis, ...] # [1, 128, 128]
        
    #     return {
    #         'input': torch.as_tensor(input_tensor, dtype=torch.float32),      # [4, 128, 128]
    #         'depth': torch.as_tensor(depth_tensor, dtype=torch.float32),      # [1, 128, 128] [新增]
    #         'target_field': torch.as_tensor(vector_field, dtype=torch.float32), 
    #         'obj_id': torch.tensor(s['obj_id'], dtype=torch.long),      
    #         'mask': torch.as_tensor(mask_tensor, dtype=torch.float32), # <--- 返回这个!
    #         'kpts_local': torch.as_tensor(kpts_local, dtype=torch.float32)      
    #     }

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
        
        # 2. BBox 处理
        bx, by, bw, bh = s['bbox']
        
        # === [核心改进] 智能 Mask 生成策略 ===
        # 步骤 A: 截取 BBox 内的深度图
        # 注意边界保护
        bx_safe, by_safe = max(0, bx), max(0, by)
        bw_safe = min(bw, img_w - bx_safe)
        bh_safe = min(bh, img_h - by_safe)
        
        depth_crop_raw = depth[by_safe:by_safe+bh_safe, bx_safe:bx_safe+bw_safe]
        
        # 步骤 B: 寻找物体的“基准深度”
        # 策略：取 BBox 中心一小块区域的中位数，避免取到边缘的噪点
        if depth_crop_raw.size > 0:
            # 过滤掉 0 值 (无效深度)
            valid_depths = depth_crop_raw[depth_crop_raw > 0]
            if valid_depths.size > 0:
                # 取中位数作为物体主体深度
                z_ref = np.median(valid_depths)
                
                # 步骤 C: 定义深度范围 (Thresholding)
                # YCB 物体直径一般在 20cm (2000单位) 以内
                # 我们取 +/- 15cm 作为容差范围，既能包住物体，又能剔除远的背景
                depth_tolerance = 1500 # 1500 * 0.1mm = 150mm = 15cm
                
                mask_crop = (depth_crop_raw > (z_ref - depth_tolerance)) & \
                            (depth_crop_raw < (z_ref + depth_tolerance))
                
                # 转为 0/255 uint8
                mask_crop = mask_crop.astype(np.uint8) * 255
            else:
                # 如果这块全是 0，说明没深度，只好退化为全 1 Mask
                mask_crop = np.ones_like(depth_crop_raw, dtype=np.uint8) * 255
        else:
            mask_crop = np.zeros((bh_safe, bw_safe), dtype=np.uint8)

        # 步骤 D: 把 crop 的 mask 放回全图尺寸 (为了后面统一 crop/resize 流程)
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        full_mask[by_safe:by_safe+bh_safe, bx_safe:bx_safe+bw_safe] = mask_crop
        # ==========================================

        # 3. 统一 Padding 和 最终裁剪
        # 使用你之前的 _pad_bbox 函数获取带 padding 的框
        x, y, w, h = self._pad_bbox(s['bbox'], img_w, img_h)
        
        crop_rgb = rgb[y:y+h, x:x+w]
        crop_mts = mts[y:y+h, x:x+w]
        crop_depth = depth[y:y+h, x:x+w]
        crop_mask = full_mask[y:y+h, x:x+w] # 裁剪刚才生成的智能 Mask
        
        # 4. Resize (Mask 和 Depth 必须 Nearest)
        crop_rgb = cv2.resize(crop_rgb, self.target_size)
        crop_mts = cv2.resize(crop_mts, self.target_size)
        crop_depth = cv2.resize(crop_depth, self.target_size, interpolation=cv2.INTER_NEAREST)
        crop_mask = cv2.resize(crop_mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # 5. 生成向量场 和 Tensor 化
        kpts_orig = np.load(s['npy_path'])
        kpts_local = kpts_orig.copy()
        scale_x = self.target_size[0] / max(w, 1)
        scale_y = self.target_size[1] / max(h, 1)
        kpts_local[:, 0] = (kpts_orig[:, 0] - x) * scale_x
        kpts_local[:, 1] = (kpts_orig[:, 1] - y) * scale_y
        
        vector_field = self._generate_vector_field(kpts_local)
        
        input_tensor = np.concatenate([
            crop_rgb.transpose(2, 0, 1) / 255.0, 
            crop_mts[np.newaxis, ...] / 255.0
        ], axis=0) 
        
        # 深度单位转换 0.1mm -> m
        depth_tensor = crop_depth.astype(np.float32) / 10000.0
        depth_tensor = depth_tensor[np.newaxis, ...] 
        
        # Mask 转 float 0/1
        mask_tensor = (crop_mask > 128).astype(np.float32)
        mask_tensor = mask_tensor[np.newaxis, ...]

        return {
            'input': torch.as_tensor(input_tensor, dtype=torch.float32),
            'depth': torch.as_tensor(depth_tensor, dtype=torch.float32),
            'target_field': torch.as_tensor(vector_field, dtype=torch.float32),
            'mask': torch.as_tensor(mask_tensor, dtype=torch.float32),
            'obj_id': torch.tensor(s['obj_id'], dtype=torch.long),
            'kpts_local': torch.as_tensor(kpts_local, dtype=torch.float32)
        }