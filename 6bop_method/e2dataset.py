import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
from pathlib import Path

#加入了template参数的dataset
class GMGPoseDataset(Dataset):
    def __init__(self, processed_dir, dataset_root, target_size=(128, 128), num_points=1024, mode='train'):
        self.processed_dir = Path(processed_dir)
        self.dataset_root = Path(dataset_root)
        self.target_size = target_size
        self.num_points = num_points # [新增] 点云采样数量，默认1024
        self.mode = mode
        
        # 预先扫描所有样本
        self.sample_list = self._build_sample_list()
        print(f"Dataset loaded: {len(self.sample_list)} samples found in {mode} mode.")
        
    def _get_random_template(self, obj_id, scene_id):
        """
        随机获取一张该物体的模板图。
        按照以下策略进行：
        1. 优先取同一场景(scene_id)下的模板（如果有），这叫 "Intra-sequence"
        2. 如果没有，就取该 obj_id 下的任意一张，这叫 "General"
        """
        # 模板根目录: processed_data/templates/obj_{id}
        tpl_dir = self.processed_dir / "templates" / f"obj_{obj_id}"
        
        if not tpl_dir.exists():
            # 容错：如果没有模板，返回全黑图
            return np.zeros((128, 128, 3), dtype=np.uint8)
        
        # 获取所有 rgb 模板
        all_tpls = list(tpl_dir.glob("*_rgb.png"))
        if not all_tpls:
            return np.zeros((128, 128, 3), dtype=np.uint8)
            
        # 随机选一张
        choice = np.random.choice(all_tpls)
        img = cv2.imread(str(choice))
        if img is None: return np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Resize 到标准大小
        img = cv2.resize(img, self.target_size)
        return img
        
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

                # 获取 Pose 信息 (R, t)
                # YCB GT 格式: "cam_R_m2c": [9个float], "cam_t_m2c": [3个float]
                gt_item = gt_data[str_key][ins_idx]
                R = np.array(gt_item["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
                t = np.array(gt_item["cam_t_m2c"], dtype=np.float32) # 单位通常是 mm
                
                info_item = info_data[str_key][ins_idx]
                bbox = info_item["bbox_visib"] 
                visib_fract = info_item.get("visib_fract", 0.0)
                
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
                    'bbox': bbox,
                    'pose_R': R,
                    'pose_t': t
                })
        
        print(f"过滤掉了 {skip_count} 个无效/低可见度样本。")
        return samples


    # === [新增] MTS 转 点云 核心函数 ===
    def _mts_to_pointcloud(self, mts_crop):
        """
        从裁剪后的 MTS 图像中采样点云。
        mts_crop: [128, 128] uint8 图像
        return: [N, 4] 点云张量 (x, y, t, p)
        """
        # 1. 找到所有非零像素 (有事件发生的地方)
        # y_idxs, x_idxs 是像素坐标
        y_idxs, x_idxs = np.where(mts_crop > 10) # 阈值10过滤掉极暗的噪点
        
        num_valid = len(y_idxs)
        N = self.num_points # 目标点数，例如 1024
        
        if num_valid > 0:
            # 2. 随机采样 N 个点
            if num_valid >= N:
                # 如果点够多，不放回采样
                choice_idx = np.random.choice(num_valid, N, replace=False)
            else:
                # 如果点不够，允许重复采样 (补齐)
                choice_idx = np.random.choice(num_valid, N, replace=True)
            
            xs = x_idxs[choice_idx].astype(np.float32)
            ys = y_idxs[choice_idx].astype(np.float32)
            
            # 3. 构造特征
            # MTS 的像素值本质上编码了时间 (Time/Intensity)
            # 我们把像素值归一化作为 't' 特征
            ts = mts_crop[y_idxs[choice_idx], x_idxs[choice_idx]].astype(np.float32) / 255.0
            
            # 'p' (Polarity) 在 MTS 里丢失了，我们可以用 0.5 填充，或者用 ts 代替
            ps = np.ones_like(ts) * 0.5 
            
            # 堆叠成 [N, 4] -> (x, y, t, p)
            # 注意：x, y 必须对应 128x128 的坐标系
            points = np.stack([xs, ys, ts, ps], axis=1)
            
        else:
            # 极端情况：MTS 全黑
            # 返回随机噪声点，防止网络报错，或者全0
            points = np.zeros((N, 4), dtype=np.float32)
            
        return points
    
    def _pad_bbox(self, bbox, img_w, img_h, padding_ratio=0.15):
        x, y, w, h = bbox
        
        # 1. 计算 Padding (保持浮点精度计算，最后再取整)
        pad_w = w * padding_ratio
        pad_h = h * padding_ratio
        
        # 2. 计算新的左上角 (x1, y1) 和 右下角 (x2, y2)
        # 这里的 int() 极其重要，它决定了切片的物理起始点
        x1 = int(max(0, x - pad_w))
        y1 = int(max(0, y - pad_h))
        
        # 计算右下角，同时确保不超出图像边界
        # 注意：这里用 x+w+pad_w 计算右下角，而不是简单的 w+2*pad
        x2 = int(min(img_w, x + w + pad_w))
        y2 = int(min(img_h, y + h + pad_h))
        
        # 3. 算出实际裁剪的宽高
        w_new = x2 - x1
        h_new = y2 - y1
        
        # 返回的是严格的整数，用于切片和坐标变换
        return [x1, y1, w_new, h_new]

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
        
       # === [核心改进] 智能 Mask 生成策略 (Otsu + 连通域筛选) ===
        bx_safe, by_safe = max(0, bx), max(0, by)
        bw_safe = min(bw, img_w - bx_safe)
        bh_safe = min(bh, img_h - by_safe)
        
        depth_crop_raw = depth[by_safe:by_safe+bh_safe, bx_safe:bx_safe+bw_safe]
        mask_crop = np.zeros_like(depth_crop_raw, dtype=np.uint8)
        
        if depth_crop_raw.size > 0:
            valid_mask = depth_crop_raw > 0
            valid_pixels = depth_crop_raw[valid_mask]
            
            if valid_pixels.size > 0:
                d_min, d_max = valid_pixels.min(), valid_pixels.max()
                
                # A. Otsu 初步分割
                if d_max - d_min > 5:
                    norm_depth = (depth_crop_raw.astype(np.float32) - d_min) / (d_max - d_min + 1e-6) * 255
                    norm_depth = norm_depth.astype(np.uint8)
                    _, otsu_mask = cv2.threshold(norm_depth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    
                    # 剔除无效深度区域
                    otsu_mask[~valid_mask] = 0
                    
                    # === [新增] B. 形态学去噪 ===
                    # 开运算：先腐蚀后膨胀，去除小白点，断开粘连
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    clean_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel)
                    
                    # === [新增] C. 连通域筛选 (只保留中心物体) ===
                    # 找出所有独立的白色块
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
                    
                    # stats: [x, y, width, height, area]
                    # centroids: [cx, cy]
                    
                    if num_labels > 1: # label 0 是背景，所以大于1才有前景
                        # 目标：找到 "主要" 物体。
                        # 评判标准：面积大 + 距离 BBox 中心近
                        
                        box_center_x = bw_safe / 2.0
                        box_center_y = bh_safe / 2.0
                        
                        best_label = -1
                        max_score = -1.0
                        
                        for i in range(1, num_labels): # 遍历所有前景块
                            area = stats[i, cv2.CC_STAT_AREA]
                            
                            # 忽略太小的噪点 (例如小于 10 个像素)
                            if area < 10: continue
                            
                            cx, cy = centroids[i]
                            
                            # 计算距离中心的距离 (归一化到 0-1)
                            dist_x = abs(cx - box_center_x) / box_center_x
                            dist_y = abs(cy - box_center_y) / box_center_y
                            dist_score = 1.0 / (dist_x + dist_y + 0.1) # 距离越近分越高
                            
                            # 综合分数 = 面积 * 距离权重
                            # 这里我们假设物体通常占据 BBox 的主体，所以面积权重很大
                            score = area * dist_score
                            
                            if score > max_score:
                                max_score = score
                                best_label = i
                        
                        if best_label != -1:
                            # 只保留选中的那个块
                            mask_crop = (labels == best_label).astype(np.uint8) * 255
                        else:
                            # 没找到合适的块，回退到原始 otsu
                            mask_crop = clean_mask
                    else:
                        mask_crop = clean_mask
                else:
                    # 深度差异极小，认为是平面物体，取全部有效区域
                    mask_crop[valid_mask] = 255
        
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        full_mask[by_safe:by_safe+bh_safe, bx_safe:bx_safe+bw_safe] = mask_crop
        # ===============================================
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

        
        
        # === [新增] 5. 生成点云 (从 Resize 后的 MTS 图中采样) ===
        # 这样采样的点坐标 (x, y) 直接对应 128x128 空间，无需再做坐标变换！
        event_points = self._mts_to_pointcloud(crop_mts)
        # ====================================================


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

         # 8. 构造 Pose 矩阵 (4x4)
        pose_gt = np.eye(4, dtype=np.float32)
        pose_gt[:3, :3] = s['pose_R']
        pose_gt[:3, 3] = s['pose_t']

        # === [新增] 读取 Template ===
        template_img = self._get_random_template(s['obj_id'], s['scene_id'])
        
        # 转 Tensor (3, 128, 128)
        template_tensor = template_img.transpose(2, 0, 1) / 255.0
        # ===========================

    
        return {
            'input': torch.as_tensor(input_tensor, dtype=torch.float32),
            'depth': torch.as_tensor(depth_tensor, dtype=torch.float32),
            'target_field': torch.as_tensor(vector_field, dtype=torch.float32),
            'mask': torch.as_tensor(mask_tensor, dtype=torch.float32),
            'obj_id': torch.tensor(s['obj_id'], dtype=torch.long),
            'kpts_local': torch.as_tensor(kpts_local, dtype=torch.float32),
            'bbox': torch.tensor(s['bbox'], dtype=torch.float32), # 用于还原坐标
            'rgb_path': str(s['rgb_path']), # 用于读取原图可视化，转为str防止路径对象报错,
            'event_points': torch.as_tensor(event_points, dtype=torch.float32),
            # === [核心修复] 返回还原参数和真值 Pose ===
            'scale': torch.tensor([scale_x, scale_y], dtype=torch.float32),
            'offset': torch.tensor([x, y], dtype=torch.float32),
            'pose_gt': torch.tensor(pose_gt, dtype=torch.float32),
             # [新增] 返回模板
            'template': torch.as_tensor(template_tensor, dtype=torch.float32)

        }