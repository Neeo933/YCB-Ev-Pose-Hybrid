import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================= ⚙️ 配置区域 =================
DATA_ROOT = "../ycb_ev_data/dataset/test_pbr" 
TARGET_OBJ_ID = 1      # 👈 重要：设置你要识别的特定物体 ID (对应 scene_gt 中的 obj_id)

# 超参数
BATCH_SIZE = 32
LR = 2e-4
EPOCHS = 20
LAMBDA_ROT = 10.0      # 旋转损失权重
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./results_refined"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================================

class YCBEventDataset(Dataset):
    def __init__(self, root_dir, target_obj_id, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_obj_id = target_obj_id
        self.data_list = []

        obj_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))
        
        print(f"正在扫描数据集，目标物体 ID: {target_obj_id}...")
        for obj_dir in obj_dirs:
            if not os.path.isdir(obj_dir): continue
            
            gt_path = os.path.join(obj_dir, "scene_gt.json")
            if not os.path.exists(gt_path): continue
            
            with open(gt_path, 'r') as f:
                scene_gt = json.load(f)
            
            for frame_id_str, instances in scene_gt.items():
                # 👈 修改点：遍历列表，找到匹配 target_obj_id 的实例
                target_instance = None
                for inst in instances:
                    if inst['obj_id'] == self.target_obj_id:
                        target_instance = inst
                        break
                
                if target_instance is None:
                    continue # 如果这一帧里没有我们要的物体，跳过

                img_name = f"{int(frame_id_str):06d}.png"
                img_path = os.path.join(obj_dir, "ev_histogram", img_name)
                
                if os.path.exists(img_path):
                    cam_R = np.array(target_instance['cam_R_m2c']).reshape(3, 3)
                    cam_t = np.array(target_instance['cam_t_m2c'])
                    
                    self.data_list.append({
                        'path': img_path,
                        'R': cam_R,
                        't': cam_t
                    })

        print(f"✅ 加载完成！在数据集中找到 {len(self.data_list)} 个物体 ID 为 {target_obj_id} 的样本。")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image = Image.open(item['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # 平移: mm -> m
        t_norm = torch.tensor(item['t'] / 1000.0, dtype=torch.float32)
        
        # 旋转: Matrix -> Quaternion [qx, qy, qz, qw]
        quat = R.from_matrix(item['R']).as_quat() 
        q_norm = torch.tensor(quat, dtype=torch.float32)
        
        label = torch.cat((t_norm, q_norm), dim=0)
        return image, label

# ================= 损失函数改进 =================
class PoseLoss(nn.Module):
    def __init__(self, lambda_rot=10.0):
        super(PoseLoss, self).__init__()
        self.lambda_rot = lambda_rot
        self.mse = nn.MSELoss()

    forward_doc = """
    pred/target: [B, 7] -> [tx, ty, tz, qx, qy, qz, qw]
    """
    def forward(self, pred, target):
        # 1. 位置损失 (MSE)
        loss_t = self.mse(pred[:, :3], target[:, :3])
        
        # 2. 旋转损失 (处理四元数双重覆盖: 1 - |<q1, q2>|)
        q1 = F.normalize(pred[:, 3:], dim=1) # 强行归一化预测值
        q2 = target[:, 3:]                   # 真值通常已经是归一化的
        
        # 计算点积的绝对值
        inner_product = torch.abs(torch.sum(q1 * q2, dim=1))
        loss_q = torch.mean(1 - inner_product)
        
        return loss_t + self.lambda_rot * loss_q, loss_t, loss_q

# ================= 训练与评估 =================

def get_model():
    # 针对事件相机，建议如果不使用预训练权重，可设置 weights=None
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7) 
    return model

def calculate_metrics(pred, target):
    # 位置误差 (cm)
    pos_err = torch.norm(pred[:, :3] - target[:, :3], dim=1).mean() * 100.0
    
    # 角度误差 (deg)
    q1 = F.normalize(pred[:, 3:], dim=1)
    q2 = F.normalize(target[:, 3:], dim=1)
    dot = torch.abs(torch.sum(q1 * q2, dim=1)).clamp(-1.0, 1.0)
    angle_err = torch.rad2deg(2 * torch.acos(dot)).mean()
    
    return pos_err.item(), angle_err.item()

def main():
    # 👈 修改点：不再使用 ImageNet 均值，改为简单的 [0,1] 缩放
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 建议先注释掉
    ])

    dataset = YCBEventDataset(DATA_ROOT, TARGET_OBJ_ID, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=20)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=20)

    model = get_model().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = PoseLoss(lambda_rot=LAMBDA_ROT)

    best_err = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss_total = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss, lt, lq = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()

        # Validation
        model.eval()
        val_pos_errs, val_rot_errs = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs)
                pe, re = calculate_metrics(preds, labels)
                val_pos_errs.append(pe)
                val_rot_errs.append(re)

        avg_pe = np.mean(val_pos_errs)
        avg_re = np.mean(val_rot_errs)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss_total/len(train_loader):.4f}")
        print(f" >>> Validation -> Pos Err: {avg_pe:.2f} cm, Rot Err: {avg_re:.2f} deg")

        if avg_pe < best_err:
            best_err = avg_pe
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))

if __name__ == "__main__":
    main()