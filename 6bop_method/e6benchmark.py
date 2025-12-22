import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
from scipy.spatial import cKDTree
import glob
from scipy.spatial import cKDTree  

# 引入你的模块
from f2dataset import GMGPoseDataset
from d3model import GMGPVNet

# ================= 配置 =================
CONFIG = {
    "model_path": "./cloudcheckpoint1222v5/with_points_single_obj13/last.pth", 
    "processed_dir": "../dataset/processed_data",
    "dataset_root": "../dataset/test_pbr",
    "model_mesh_root": None, # 如果有 .ply 文件填这里
    "target_obj_id": None, # 必须和训练时一致

    
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "ref_size": 70.0, 
    "cam_K": np.array([
        [1066.778, 0.0, 312.9869],
        [0.0, 1067.487, 241.3109],
        [0.0, 0.0, 1.0]
    ]),
    "num_eval_samples": 2000,
    "vis_interval": 50, # 每隔多少帧保存一张可视化图
    "vis_dir": "./benchmark_vis/1223" # 可视化保存路径
}
class Visualizer:
    def __init__(self, cam_K):
        self.K = cam_K

    def draw_cuboid(self, img, rvec, tvec, box_pts_3d, color, thickness=2):
        """画平滑的 3D 包围盒"""
        # 投影 8 个角点
        img_pts, _ = cv2.projectPoints(box_pts_3d[:8], rvec, tvec, self.K, None)
        img_pts = img_pts.squeeze().astype(int)
        
        # 12 条棱的连接关系
        edges = [
            (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), 
            (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
        ]
        
        h, w = img.shape[:2]
        
        # 绘制
        for s, e in edges:
            # 简单的边界检查，防止画飞出屏幕报错
            if self._is_in_image(img_pts[s], w, h) or self._is_in_image(img_pts[e], w, h):
                cv2.line(img, tuple(img_pts[s]), tuple(img_pts[e]), color, thickness, cv2.LINE_AA)
        return img

    def draw_axes(self, img, rvec, tvec, length=50):
        """画 RGB 坐标轴 (红X, 绿Y, 蓝Z)"""
        center_3d = np.array([[0,0,0]], dtype=np.float32)
        axis_3d = np.array([[length,0,0], [0,length,0], [0,0,length]], dtype=np.float32)
        
        # 投影
        center_2d, _ = cv2.projectPoints(center_3d, rvec, tvec, self.K, None)
        axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, self.K, None)
        
        c = tuple(center_2d.squeeze().astype(int))
        axis = axis_2d.squeeze().astype(int)
        
        # BGR 颜色: Z=红(OpenCV里通常反着来，这里我们按 RGB=XYZ 对应 BGR=ZYX)
        # X轴 (红)
        cv2.line(img, c, tuple(axis[0]), (0, 0, 255), 3, cv2.LINE_AA)
        # Y轴 (绿)
        cv2.line(img, c, tuple(axis[1]), (0, 255, 0), 3, cv2.LINE_AA)
        # Z轴 (蓝)
        cv2.line(img, c, tuple(axis[2]), (255, 0, 0), 3, cv2.LINE_AA)
        return img

    def _is_in_image(self, pt, w, h):
        return 0 <= pt[0] < w and 0 <= pt[1] < h

    def create_comparison_image(self, img_path, r_gt, t_gt, r_pred, t_pred, box_pts, error_val):
        """生成对比图：左边 GT，右边 Pred"""
        img_raw = cv2.imread(img_path)
        if img_raw is None: return None
        
        # 1. 绘制 GT (左图) - 蓝色框 (255, 0, 0)
        img_gt = img_raw.copy()
        self.draw_cuboid(img_gt, r_gt, t_gt, box_pts, (255, 100, 0), 2) # 蓝色偏深
        cv2.putText(img_gt, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 2. 绘制 Pred (右图) - 绿色框 (0, 255, 0) + 坐标轴
        img_pred = img_raw.copy()
        self.draw_cuboid(img_pred, r_pred, t_pred, box_pts, (0, 255, 0), 2)
        # 加上坐标轴显得更专业
        self.draw_axes(img_pred, r_pred, t_pred, length=40)
        
        # 写上误差
        info_text = f"ADD Error: {error_val:.1f} mm"
        color = (0, 255, 0) if error_val < 20 else (0, 0, 255) # 误差小绿字，大红字
        cv2.putText(img_pred, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 3. 拼接
        combined = np.hstack([img_gt, img_pred])
        return combined
class BenchmarkRunner:
    def __init__(self):
        self.device = CONFIG["device"]
        os.makedirs(CONFIG["vis_dir"], exist_ok=True)
        
        print(f"Loading Model: {CONFIG['model_path']}")
        self.model = GMGPVNet(num_keypoints=9).to(self.device)
        self.model.load_state_dict(torch.load(CONFIG["model_path"], map_location=self.device))
        self.model.eval()
        
        self.meshes = {} 
        
        # 3D 关键点定义 (必须与 DataFactory 保持一致，中心点在最后)
        s = CONFIG["ref_size"]
        self.box_pts_3d = np.array([
            [s,s,s], [s,s,-s], [s,-s,s], [s,-s,-s],
            [-s,s,s], [-s,s,-s], [-s,-s,s], [-s,-s,-s],
            [0,0,0] # Center at index 8
        ], dtype=np.float32)

    def load_mesh_points(self, obj_id):
        """加载 Mesh 用于计算 ADD"""
        if obj_id in self.meshes: return self.meshes[obj_id]
        
        # 降级方案：随机采样
        s = CONFIG["ref_size"]
        dummy_pts = np.random.uniform(-s, s, (500, 3)).astype(np.float32)
        self.meshes[obj_id] = dummy_pts
        return dummy_pts

    # def get_voting_kpts(self, pred_vec, pred_mask):
    #     """RANSAC / WLS 投票"""
    #     c, h, w = pred_vec.shape
    #     mask = torch.sigmoid(pred_mask[0]) > 0.9
    #     y_idxs, x_idxs = torch.where(mask)
        
    #     if len(x_idxs) < 10: return None # 像素太少，视为检测失败

    #     coords = torch.stack([x_idxs, y_idxs], dim=1).float().cpu().numpy()
    #     vectors = pred_vec[:, mask].cpu().numpy().T 
        
    #     kpts_pred = []
    #     for k in range(9):
    #         vx = vectors[:, k*2]
    #         vy = vectors[:, k*2+1]
    #         A = np.stack([vy, -vx], axis=1)
    #         b = vy * coords[:, 0] - vx * coords[:, 1]
    #         res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    #         kpts_pred.append(res)
            
    #     return np.array(kpts_pred)
    def get_voting_kpts(self, pred_vec, pred_mask):
        """
        RANSAC 投票：抗噪能力更强
        """
        c, h, w = pred_vec.shape
        # 提高 Mask 阈值，只取最可信的像素
        mask = torch.sigmoid(pred_mask[0]) > 0.5
        y_idxs, x_idxs = torch.where(mask)
        
        # 点太少直接放弃
        if len(x_idxs) < 30: return None

        coords = torch.stack([x_idxs, y_idxs], dim=1).float().cpu().numpy() # [N, 2]
        vectors = pred_vec[:, mask].cpu().numpy().T # [N, 18]
        
        kpts_2d = []
        
        # 对 9 个关键点分别计算
        for k in range(9):
            vx = vectors[:, k*2]
            vy = vectors[:, k*2+1]
            
            # 构造 RANSAC 需要的数据形式
            # 这里的思路是：我们在 N 个像素中，随机选 2 个点，算出它们的交点
            # 重复多次，看哪个交点被支持得最多
            
            # 为了简单高效，我们使用 OpenCV 的 RANSAC 思想
            # 但 OpenCV 没有直接针对“向量场交点”的 RANSAC 函数
            # 这里提供一个简化的“加权中位数”策略，比最小二乘鲁棒得多
            
            A = np.stack([vy, -vx], axis=1)
            b = vy * coords[:, 0] - vx * coords[:, 1]
            
            # 1. 初步解 (最小二乘)
            initial_kp, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # 2. 计算每个像素对这个解的“满意度” (Residual Error)
            # 理想向量 vs 实际向量 的余弦相似度
            vec_to_kp = initial_kp - coords
            dist = np.linalg.norm(vec_to_kp, axis=1) + 1e-6
            vec_to_kp_norm = vec_to_kp / dist[:, None]
            
            dot_prod = vec_to_kp_norm[:, 0] * vx + vec_to_kp_norm[:, 1] * vy
            
            # 3. 剔除离群点 (Inlier Selection)
            # 只有方向偏差小于一定角度 (比如 cos > 0.9) 的点才是好点
            inliers = dot_prod > 0.9
            
            if np.sum(inliers) > 10:
                # 4. 用好点再算一次 (Refinement)
                A_in = A[inliers]
                b_in = b[inliers]
                final_kp, _, _, _ = np.linalg.lstsq(A_in, b_in, rcond=None)
                kpts_2d.append(final_kp)
            else:
                # 如果好点太少，说明预测很烂，只能用初始解凑合
                kpts_2d.append(initial_kp)
            
        return np.array(kpts_2d)
    # def compute_add_metric(self, R_pred, t_pred, R_gt, t_gt, obj_id):
    #     pts = self.load_mesh_points(obj_id.item())
    #     pts_pred = (np.dot(pts, R_pred.T) + t_pred.T)
    #     pts_gt = (np.dot(pts, R_gt.T) + t_gt.T)
    #     return np.mean(np.linalg.norm(pts_pred - pts_gt, axis=1))

    def compute_add_metric(self, R_pred, t_pred, R_gt, t_gt, obj_id):
        """
        计算姿态误差：
        - 非对称物体：使用 ADD (Average Distance of Model Points)
        - 对称物体：使用 ADD-S (Average Distance of Model Points with Symmetry)
        """
        pts = self.load_mesh_points(obj_id.item())
        
        # 1. 将模型点云变换到相机坐标系
        pts_pred = (np.dot(pts, R_pred.T) + t_pred.T)
        pts_gt = (np.dot(pts, R_gt.T) + t_gt.T)
        
        # 2. 定义对称物体的 ID 列表 (YCB-Video 数据集标准)
        # ID 说明: 
        # 13: bowl (碗)
        # 16: abrasive sponge (海绵擦 - 也是几何对称的)
        # 19: pitcher base (水壶底)
        # 20: gelatin box (果冻盒 - 纹理对称)
        # 21: potted meat (罐头)
        # 24: extra_large_clamp (大夹子 - 也是几何对称)
        # 请根据你的 dataset/test_pbr 里的实际 obj_id 确认这些数字
        symmetric_ids = [13, 16, 19, 20, 21, 24,3,9] 
        
        # 3. 根据物体类型选择算法
        if int(obj_id) in symmetric_ids:
            # === ADD-S (针对对称物体) ===
            # 逻辑：对于预测点云中的每一个点，在真值点云中找一个离它最近的点算距离
            # 这样即使旋转差了 180 度（对于对称物体外观一样），误差也会很小
            kdtree = cKDTree(pts_gt)
            distances, _ = kdtree.query(pts_pred) # 返回每个点的最近邻距离
            mean_dist = np.mean(distances)
        else:
            # === ADD (针对非对称物体) ===
            # 逻辑：点对点严格对应计算距离
            mean_dist = np.mean(np.linalg.norm(pts_pred - pts_gt, axis=1))
            
        return mean_dist

    def draw_visuals(self, img_path, kpts_pred,kpts_gt, rvec, tvec, save_name):
        """绘制 2D 关键点和 3D 包围盒"""
        img = cv2.imread(img_path)
        if img is None: return

        # 1. 画 2D 预测点 (黄色)
        for kp in kpts_pred:
            cv2.circle(img, (int(kp[0]), int(kp[1])), 3, (0, 255, 255), -1)

        # 2. [新增] 画 2D 真值点 (红色 - GT)
    # 这能让你一眼看出是网络预测歪了，还是坐标系本身就歪了
        for kp in kpts_gt:
            cv2.circle(img, (int(kp[0]), int(kp[1])), 2, (0, 0, 255), -1)
            
        # 2. 画 3D 投影框 (绿色)
        # 只用前8个角点画框
        img_pts, _ = cv2.projectPoints(self.box_pts_3d[:8], rvec, tvec, CONFIG["cam_K"], None)
        img_pts = img_pts.squeeze().astype(int)
        
        # 定义立方体的 12 条棱 (基于 0-7 的索引)
        edges = [
            (0,1), (0,2), (0,4), 
            (1,3), (1,5), 
            (2,3), (2,6), 
            (3,7), 
            (4,5), (4,6), 
            (5,7), 
            (6,7)
        ]
        
        # 绘制线框
        for s, e in edges:
            # 增加边界检查防止画出图外报错
            if 0 <= s < len(img_pts) and 0 <= e < len(img_pts):
                cv2.line(img, tuple(img_pts[s]), tuple(img_pts[e]), (0, 255, 0), 2)

        cv2.imwrite(save_name, img)

    def run(self):
        dataset = GMGPoseDataset(
            processed_dir=CONFIG["processed_dir"], 
            dataset_root=CONFIG["dataset_root"],
            mode='train' ,
            target_obj_id=CONFIG["target_obj_id"] # <--- 传入这里

        )
        
        total_samples = len(dataset)
        if CONFIG["num_eval_samples"]:
            indices = np.random.choice(total_samples, CONFIG["num_eval_samples"], replace=False)
        else:
            indices = range(total_samples)
            
        print(f"Start Benchmarking on {len(indices)} samples...")
        vis_tool = Visualizer(CONFIG["cam_K"])

        # === 统计计数器 ===
        stats = {
            "total": len(indices),
            "success_10": 0,    # ADD < 0.1d
            "fail_det": 0,      # Mask 像素不足 (Detection Failed)
            "fail_pnp": 0,      # PnP 解算失败
            "fail_large_err": 0 # 解算成功但误差过大
        }
        
        add_errors = []
        diameter = 200.0 # mm
        
        for i, idx in enumerate(tqdm(indices)):
            sample = dataset[idx]
            
            inputs = sample['input'].unsqueeze(0).to(self.device)
            depth = sample['depth'].unsqueeze(0).to(self.device)
            event_points = sample['event_points'].unsqueeze(0).to(self.device) if 'event_points' in sample else None
            if 'template' in sample:
                template = sample['template'].unsqueeze(0).to(self.device)
            else:
                # 容错：如果旧版 Dataset 没返回 template，造一个全黑的
                # 但这会严重影响精度，建议务必更新 Dataset
                template = torch.zeros_like(inputs[:, :3, :, :])

            if 'event_points' in sample:
                event_points = sample['event_points'].unsqueeze(0).to(self.device)
            else:
                event_points = None

            # 1. 推理
            with torch.no_grad():
                # [修改] 传入 template
                pred_vec, pred_mask = self.model(inputs, depth, template, event_points)
     
            # 2. 投票
            kpts_crop = self.get_voting_kpts(pred_vec[0], pred_mask[0])
            
            # [统计] 检测失败
            if kpts_crop is None:
                stats["fail_det"] += 1
                add_errors.append(1000.0)
                continue
                
            # 3. 坐标还原
            # scale = sample['scale'].numpy()
            # offset = sample['offset'].numpy()
            
            # kpts_global = kpts_crop.copy()
            # kpts_global[:, 0] = kpts_crop[:, 0] / scale[0] + offset[0]
            # kpts_global[:, 1] = kpts_crop[:, 1] / scale[1] + offset[1]

            # # === [新增] 坐标还原 (GT) ===
            # # 从 Dataset 拿原始的 local gt
            # kpts_gt_local = sample['kpts_local'].numpy()
            # kpts_gt_global = kpts_gt_local.copy()
            # kpts_gt_global[:, 0] = kpts_gt_local[:, 0] / scale[0] + offset[0]
            # kpts_gt_global[:, 1] = kpts_gt_local[:, 1] / scale[1] + offset[1]
            # # ===========================

            # 3. 坐标还原
            # 直接信任 Dataset 传出来的 scale 和 offset，因为它们是训练时“案发现场”的真实参数
            scale = sample['scale'].numpy()   
            offset = sample['offset'].numpy() 
            
            kpts_global = kpts_crop.copy()
            kpts_global[:, 0] = kpts_crop[:, 0] / scale[0] + offset[0]
            kpts_global[:, 1] = kpts_crop[:, 1] / scale[1] + offset[1]
            
            # 同理还原 GT (用于画红点验证)
            kpts_gt_local = sample['kpts_local'].numpy()
            kpts_gt_global = kpts_gt_local.copy()
            kpts_gt_global[:, 0] = kpts_gt_local[:, 0] / scale[0] + offset[0]
            kpts_gt_global[:, 1] = kpts_gt_local[:, 1] / scale[1] + offset[1]

            
            # 4. PnP 解算
            ret_pred, rvec_pred, tvec_pred = cv2.solvePnP(
                self.box_pts_3d, kpts_global, CONFIG["cam_K"], None, flags=cv2.SOLVEPNP_EPNP
            )


            # === [新增] 距离保护机制 ===
            # 如果算出来的距离大于 3米 (YCB场景通常在1米左右)，说明点缩成一团了
            if tvec_pred[2] > 3000.0: 
                # 强制修正 Z 轴到 1米 (假设)
                # 这是一种 heuristic，虽然 R 还是错的，但至少 t 不会离谱
                scale_factor = 1000.0 / tvec_pred[2]
                tvec_pred = tvec_pred * scale_factor
                # 或者标记为失败
                # stats["fail_large_err"] += 1
            
            # [统计] PnP 失败
            if not ret_pred:
                stats["fail_pnp"] += 1
                add_errors.append(1000.0)
                continue

            ## 用深度图做深度
            # 1. 获取 crop 区域的深度图 (128x128)
            d_crop = depth[0, 0].cpu().numpy() # 单位是 米
            # 2. 获取预测的 Mask (只取置信度高的区域)
            m_crop = torch.sigmoid(pred_mask[0, 0]).cpu().numpy() > 0.5
            
            if np.sum(m_crop) > 10:
                # 取 Mask 区域内的深度中位数
                valid_depths = d_crop[m_crop]
                # 过滤掉 0 值
                valid_depths = valid_depths[valid_depths > 0]
                
                if len(valid_depths) > 0:
                    z_measured_m = np.median(valid_depths)
                    z_measured_mm = z_measured_m * 1000.0 # 注意：你的Dataset里除以了10000，这里要乘回来
                    
                    # 强行修正 tvec 的 Z 分量
                    # print(f"Fixing Z: PnP={tvec_pred[2][0]:.1f} -> Depth={z_measured_mm:.1f}")
                    tvec_pred[2] = z_measured_mm
            
            # 5. 算分
            R_pred, _ = cv2.Rodrigues(rvec_pred)
            pose_gt = sample['pose_gt'].numpy()
            R_gt = pose_gt[:3, :3]
            t_gt = pose_gt[:3, 3]

            # 转为 rvec 用于 opencv 画图
            rvec_gt, _ = cv2.Rodrigues(R_gt)
            tvec_gt = t_gt # shape (3,)
            
            # 确保 tvec_gt 是 float64 且 shape (3, 1) 以防万一
            tvec_gt = tvec_gt.reshape(3, 1).astype(np.float64)

            # ... (在 solvePnP 之后) ...
            
            # === [新增] Debug 诊断模块 (只打印前几个样本) ===
            if i < 3: 
                print(f"\n--- Debug Sample {idx} ---")
                print(f"GT Z:   {t_gt[2]:.2f}")
                print(f"Pred Z: {tvec_pred[2][0]:.2f} (Refined)")
                print(f"Error:  {np.linalg.norm(t_gt - tvec_pred.flatten()):.2f}")
                # 1. 检查 GT 和 Pred 的平移向量 (t)
                # 如果 GT 是 ~1000，Pred 是 ~13000，说明确实是“缩成一团”导致推得太远
                print(f"GT  tvec (mm): {t_gt.flatten()}")
                print(f"Pred tvec (mm): {tvec_pred.flatten()}")
                
                # 2. 检查 2D 关键点的分布范围 (Spread)
                # 计算 2D 点的标准差，看是不是缩成一团
                spread_x = np.std(kpts_global[:, 0])
                spread_y = np.std(kpts_global[:, 1])
                print(f"Pred 2D Spread: X_std={spread_x:.2f}, Y_std={spread_y:.2f}")
                
                # 如果 std 很小 (比如 < 5.0)，说明所有点都挤在一起 -> 模式坍塌
                if spread_x < 5.0 and spread_y < 5.0:
                    print("⚠️ 警告：预测关键点重合！模型发生了模式坍塌 (Mode Collapse)。")
                else:
                    print("✅ 2D 关键点分布正常。")

                # 3. 检查 3D 框尺寸 (Ref Size)
                # 确保 benchmark 里的 ref_size 和训练时一致
                print(f"Ref Size used: {CONFIG['ref_size']}")
            # ===============================================

            
            # 6. 算分
            error = self.compute_add_metric(R_pred, tvec_pred.reshape(3), R_gt, t_gt, sample['obj_id'])
            add_errors.append(error)
            
            if error < 0.1 * diameter:
                stats["success_10"] += 1
            else:
                stats["fail_large_err"] += 1

            # 6. 可视化 (每隔 N 帧保存一张)
            if i % CONFIG["vis_interval"] == 0:
                save_name = os.path.join(CONFIG["vis_dir"], f"eval_{i}_err{error:.1f}.jpg")
                self.draw_visuals(sample['rgb_path'], kpts_global, kpts_gt_global, rvec_pred, tvec_pred, save_name)
            # [修改] 可视化部分
            if i % CONFIG["vis_interval"] == 0:
                # 传入 GT 和 Pred 的 R, t
                # 注意：rvec 需要是旋转向量形式，如果之前转成了矩阵 R_pred，这里要转回来，或者直接用 rvec_pred
                # cv2.Rodrigues 可以互转
                
                # 确保 box_pts_3d 是针对当前物体的 (如果你做了动态大小调整)
                current_box = self.box_pts_3d # 或者 self.get_box_pts(...)
                
                vis_img = vis_tool.create_comparison_image(
                    sample['rgb_path'],
                    rvec_gt, tvec_gt,       # GT 姿态 (需要你从 pose_gt 转一下 rvec)
                    rvec_pred, tvec_pred,   # Pred 姿态
                    current_box,
                    error
                )
                
                if vis_img is not None:
                    save_name = os.path.join(CONFIG["vis_dir"], f"eval_{i}_err_improved{error:.0f}mm.jpg")
                    cv2.imwrite(save_name, vis_img)

        # 7. 打印最终报告
        accuracy = stats["success_10"] / stats["total"] * 100
        mean_error = np.mean(add_errors)
        
        print("\n" + "="*40)
        print(f"Model: {CONFIG['model_path']}")
        print(f"Total Samples: {stats['total']}")
        print("-" * 20)
        print(f"✅ Accuracy (<10% d): {accuracy:.2f}%")
        print(f"📏 Mean ADD Error:    {mean_error:.2f} mm")
        print("-" * 20)
        print("Failure Analysis:")
        print(f"❌ Detection Failed:  {stats['fail_det']} ({stats['fail_det']/stats['total']:.1%}) -> Mask too small/empty")
        print(f"❌ PnP Failed:        {stats['fail_pnp']} ({stats['fail_pnp']/stats['total']:.1%}) -> Numerical instability")
        print(f"❌ Large Error:       {stats['fail_large_err']} ({stats['fail_large_err']/stats['total']:.1%}) -> Pose inaccurate")
        print("="*40)
        print(f"Visualizations saved to: {CONFIG['vis_dir']}")

        add_errors = np.array(add_errors)
        
        # 统计不同阈值的准确率
        acc_2cm = np.mean(add_errors < 20.0) * 100
        acc_5cm = np.mean(add_errors < 50.0) * 100
        acc_10cm = np.mean(add_errors < 100.0) * 100
        
        # 计算 AUC (Area Under Curve) - 0到10cm的曲线下面积
        # 这是更科学的综合指标
        thresholds = np.linspace(0, 100, 1000) # 0到10cm
        precision = [(add_errors < t).mean() for t in thresholds]
        auc = np.trapz(precision, thresholds) / 100.0 * 100

        print("\n" + "="*40)
        print(f"Model: {CONFIG['model_path']}")
        print(f"Total Samples: {stats['total']}")
        print("-" * 20)
        print(f"✅ Acc (< 2 cm):  {acc_2cm:.2f}%  (Strict)")
        print(f"✅ Acc (< 5 cm):  {acc_5cm:.2f}%  (Coarse)")
        print(f"✅ Acc (< 10 cm): {acc_10cm:.2f}% (Robust)")
        print(f"🏆 AUC (0-10cm):  {auc:.2f}%      (Overall Performance)")
        print("-" * 20)
        print(f"📏 Mean ADD Error:    {np.mean(add_errors):.2f} mm")
        print(f"📏 Median ADD Error:  {np.median(add_errors):.2f} mm (排除离群点)")
        print("="*40)

if __name__ == "__main__":
    bencher = BenchmarkRunner()
    bencher.run()