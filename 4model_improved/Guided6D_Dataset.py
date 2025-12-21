import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os

class Guided6DDataset(Dataset):
    def __init__(self, data_root, obj_id, img_size=256):
        self.root = data_root
        self.img_size = img_size
        self.obj_id = obj_id # 针对特定物体的训练
        # 假设你有一个列表记录了所有合法的样本索引
        self.indices = self._load_indices() 

    def __len__(self):
        return len(self.indices)

    def _process_input(self, mts_img, depth_img):
        """实现你之前验证过的三通道合成逻辑"""
        h, w = depth_img.shape
        # 1. MTS 边缘提取与去重影
        b, g, r = cv2.split(mts_img)
        grad = cv2.Sobel(g, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(g, cv2.CV_32F, 0, 1)**2
        mts_chan = cv2.normalize(np.sqrt(grad), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 2. 深度处理
        depth_f = depth_img.astype(np.float32)
        depth_chan = cv2.normalize(depth_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 3. 梯度处理
        depth_grad = cv2.Sobel(depth_chan, cv2.CV_8U, 1, 1)
        
        # 合成并 Resize 到模型尺寸 (256, 256, 3)
        input_stack = cv2.merge([mts_chan, depth_grad, depth_chan])
        input_resized = cv2.resize(input_stack, (self.img_size, self.img_size))
        
        # 转换为 Tensor: [3, 256, 256]
        return torch.from_numpy(input_resized).permute(2, 0, 1).float() / 255.0

    def _generate_gt_xyz(self, pose_label, camera_k):
        """
        核心任务：根据位姿生成 GT XYZ 坐标图
        这里需要加载物体的 3D 模型点云，将其按 pose 变换后投影
        """
        # 简化演示：返回一个与输出尺寸一致的随机坐标图 [3, 32, 32]
        # 在实际中，你需要利用渲染器或者反投影公式生成
        return torch.randn(3, 32, 32)

    def __getitem__(self, idx):
        # 1. 加载数据
        idx_str = self.indices[idx]
        mts_path = os.path.join(self.root, f"rgb_events/{idx_str}.png")
        depth_path = os.path.join(self.root, f"depth/{idx_str}.png")
        
        mts_img = cv2.imread(mts_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        # 2. 生成输入张量
        input_tensor = self._process_input(mts_img, depth_img)

        # 3. 生成标签 (XYZ Map & Mask)
        gt_xyz = self._generate_gt_xyz(None, None)
        gt_mask = (torch.sum(input_tensor, dim=0, keepdim=True) > 0).float()
        # 将 Mask 下采样到与输出一致 (32x32)
        gt_mask_small = torch.nn.functional.interpolate(gt_mask.unsqueeze(0), size=(32, 32)).squeeze(0)

        return input_tensor, gt_xyz, gt_mask_small

# --- 3. 训练循环 (Training Loop) 原型 ---
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    for batch_idx, (data, target_xyz, target_mask) in enumerate(dataloader):
        data, target_xyz, target_mask = data.to(device), target_xyz.to(device), target_mask.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # 损失函数设计
        # 1. XYZ 回归损失 (只计算前景部分的 L1 Loss)
        loss_xyz = F.l1_loss(output['xyz_map'] * target_mask, target_xyz * target_mask)
        
        # 2. Mask 分割损失
        loss_mask = F.binary_cross_entropy_with_logits(output['mask_logit'], target_mask)
        
        total_loss = loss_xyz + 1.0 * loss_mask
        total_loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Loss = {total_loss.item():.4f}")

# --- 4. 运行准备 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Guided6DNet().to(device)
dataset = Guided6DDataset(data_root="../ycb_ev_data/dataset/test_pbr/000000", obj_id=1)
loader = DataLoader(dataset, batch_size=16, shuffle=True)