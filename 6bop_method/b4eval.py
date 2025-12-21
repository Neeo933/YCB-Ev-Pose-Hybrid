import torch
import numpy as np
import cv2
import json
import os
from pathlib import Path
from tqdm import tqdm

# 引入你的模块
from a2dataset import GMGPoseDataset
from a3model import GMGPVNet

# ================= 配置 =================
CONFIG = {
    # 你的模型权重路径 (选 best.pth)
    "model_path": "./checkpointsv4/best.pth", 
    "processed_dir": "../dataset/processed_data",
    "dataset_root": "../dataset/test_pbr",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # 3D 框尺寸 (必须与 DataFactory 中的 self.ref_size 一致!)
    "ref_size": 50.0, 
    
    # 相机内参 (YCB-PBR 的标准内参，如有变动请修改)
    "cam_K": np.array([
        [1066.778, 0.0, 312.9869],
        [0.0, 1067.487, 241.3109],
        [0.0, 0.0, 1.0]
    ]),
    
    "output_dir": "./eval_results"
}

class PoseEvaluator:
    def __init__(self):
        self.device = CONFIG["device"]
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        
        # 1. 加载模型
        print(f"正在加载模型: {CONFIG['model_path']}")
        self.model = GMGPVNet(num_keypoints=9).to(self.device)
        # 加载权重 (处理可能的 strict=False 情况)
        state_dict = torch.load(CONFIG["model_path"], map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # 2. 定义 3D 物体点 (8角点 + 1中心)
        s = CONFIG["ref_size"]
        self.object_pts_3d = np.array([
            [s,s,s], [s,s,-s], [s,-s,s], [s,-s,-s],
            [-s,s,s], [-s,s,-s], [-s,-s,s], [-s,-s,-s],
            [0,0,0]
        ], dtype=np.float32)

    def voting_layer(self, pred_vectors, pred_mask):
        """
        简单版最小二乘法投票
        pred_vectors: [18, 128, 128]
        pred_mask: [1, 128, 128]
        Return: [9, 2] Crop坐标系下的关键点
        """
        c, h, w = pred_vectors.shape
        
        # 1. 筛选投票像素 (Mask > 0.5)
        mask = torch.sigmoid(pred_mask[0]) > 0.5
        y_idxs, x_idxs = torch.where(mask)
        
        # 如果有效像素太少，返回 None
        if len(x_idxs) < 10: return None

        coords = torch.stack([x_idxs, y_idxs], dim=1).float().to(self.device) # [N, 2]
        kpts = []

        # 2. 对 9 个关键点分别投票
        for k in range(9):
            # 获取向量
            vx = pred_vectors[k*2][mask]
            vy = pred_vectors[k*2+1][mask]
            
            # 最小二乘求解: A x = b
            # A = [vy, -vx], b = vy*px - vx*py
            A = torch.stack([vy, -vx], dim=1)
            b = vy * coords[:, 0] - vx * coords[:, 1]
            
            # PyTorch 的 lstsq (或者用 inverse)
            # (A^T A)^-1 A^T b
            AtA = A.T @ A
            Atb = A.T @ b
            
            try:
                # 加上微小扰动防止奇异矩阵
                det = torch.det(AtA)
                if det.abs() < 1e-6: 
                    # 如果这都能奇异，说明所有向量都平行，直接取均值糊弄一下
                    kp = coords.mean(dim=0)
                else:
                    kp = torch.linalg.solve(AtA, Atb)
            except:
                kp = coords.mean(dim=0)
                
            kpts.append(kp)
            
        return torch.stack(kpts).cpu().numpy() # [9, 2]

    def draw_3d_box(self, img, rvec, tvec):
        """在图上画 3D 绿框"""
        # 投影 3D 点到 2D
        img_pts, _ = cv2.projectPoints(self.object_pts_3d[:8], rvec, tvec, CONFIG["cam_K"], None)
        img_pts = img_pts.squeeze().astype(int)
        
        # 绘制 12 条棱
        edges = [
            (0,1), (1,3), (3,2), (2,0),
            (4,5), (5,7), (7,6), (6,4),
            (0,4), (1,5), (2,6), (3,7)
        ]
        
        img_vis = img.copy()
        for s, e in edges:
            cv2.line(img_vis, tuple(img_pts[s]), tuple(img_pts[e]), (0, 255, 0), 2)
            
        return img_vis

    def run_demo(self, num_samples=5):
        # 1. 准备数据 (使用 Dataset 直接读取，方便拿元数据)
        dataset = GMGPoseDataset(
            processed_dir=CONFIG["processed_dir"], 
            dataset_root=CONFIG["dataset_root"],
            target_size=(128, 128),
            mode='train' 
        )
        
        print(f"开始推理 {num_samples} 个样本...")
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        for idx in indices:
            # 直接从 dataset 拿 sample (这是一个字典)
            sample = dataset[idx]
            
            # 转为 Batch 形式 (增加 batch 维度 [1, ...])
            inputs = sample['input'].unsqueeze(0).to(self.device)
            depth = sample['depth'].unsqueeze(0).to(self.device)
            
            # [修复] bbox 从 tensor 转回 numpy
            bbox = sample['bbox'].numpy() # [x, y, w, h]
            
            # --- Inference ---
            with torch.no_grad():
                pred_vec, pred_mask = self.model(inputs, depth, event_points=None)

            # ================= [新增] 向量场诊断 (保存图片版) =================
            # 目的：检查 Channel 0 (角点) 和 Channel 16 (中心) 是否长得一样
            
            # 1. 提取数据 (只取 X 分量)
            # pred_vec: [1, 18, 128, 128]
            vec_corner = pred_vec[0, 0].cpu().numpy()  # 指向第1个角点
            vec_center = pred_vec[0, 16].cpu().numpy() # 指向中心点
            
            # 2. 归一化并转伪彩色 (Heatmap)
            def process_heatmap(v, label):
                # 归一化到 0-255
                v_min, v_max = v.min(), v.max()
                v_norm = (v - v_min) / (v_max - v_min + 1e-6)
                v_uint8 = (v_norm * 255).astype(np.uint8)
                
                #以此生成伪彩色图 (Jet: 蓝->红 代表 数值小->大)
                heatmap = cv2.applyColorMap(v_uint8, cv2.COLORMAP_JET)
                
                # 打上标签文字
                cv2.putText(heatmap, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.4, (255, 255, 255), 1)
                cv2.putText(heatmap, f"Range:[{v_min:.2f}, {v_max:.2f}]", (5, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                return heatmap

            hm_corner = process_heatmap(vec_corner, "To Corner (Ch0)")
            hm_center = process_heatmap(vec_center, "To Center (Ch16)")
            
            # 3. 拼接到一起 (左右拼接)
            debug_img = np.hstack([hm_corner, hm_center])
            
            # 4. 保存诊断图
            debug_path = str(Path(CONFIG["output_dir"]) / f"debug_vec_{idx}_obj{sample['obj_id']}.png")
            cv2.imwrite(debug_path, debug_img)
            print(f"  -> [Debug] 向量场诊断图已保存: {debug_path}")
            # ==============================================================
            
            # --- Voting (Crop Space) ---
            kpts_crop = self.voting_layer(pred_vec[0], pred_mask[0])
            
            # [修复] 读取原图路径
            orig_path = sample['rgb_path']
            
            if kpts_crop is None:
                print(f"Sample {idx}: 物体检测失败 (Mask空)")
                continue

            # --- Coordinate Un-crop (Crop -> Global) ---
            # 重新计算 padding 后的框 (必须和训练时逻辑一致)
            # 我们需要知道原图尺寸，临时读一下图片头或者硬编码 640x480
            orig_img = cv2.imread(orig_path)
            if orig_img is None: 
                print(f"无法读取原图: {orig_path}")
                continue
            
            H_full, W_full = orig_img.shape[:2]
            
            # 调用 dataset 的辅助函数算 pad 后的框
            x, y, w, h = dataset._pad_bbox(bbox, W_full, H_full)
            
            scale_x = 128.0 / w
            scale_y = 128.0 / h
            
            kpts_global = kpts_crop.copy()
            kpts_global[:, 0] = kpts_crop[:, 0] / scale_x + x
            kpts_global[:, 1] = kpts_crop[:, 1] / scale_y + y
            
            # --- PnP Solver ---
            success, rvec, tvec = cv2.solvePnP(
                self.object_pts_3d, 
                kpts_global, 
                CONFIG["cam_K"], 
                distCoeffs=None,
                flags=cv2.SOLVEPNP_EPNP
            )
            
            if not success:
                print(f"Sample {idx}: PnP 求解失败")
                continue

            # --- Visualization ---
            res_img = self.draw_3d_box(orig_img, rvec, tvec)
            
            # 画投票出来的点 (黄色)
            for kp in kpts_global:
                cv2.circle(res_img, (int(kp[0]), int(kp[1])), 3, (0, 255, 255), -1)

            save_name = str(Path(CONFIG["output_dir"]) / f"res_{idx}_obj{sample['obj_id']}.png")
            cv2.imwrite(save_name, res_img)
            print(f"Saved: {save_name}")

if __name__ == "__main__":
    evaluator = PoseEvaluator()
    evaluator.run_demo()