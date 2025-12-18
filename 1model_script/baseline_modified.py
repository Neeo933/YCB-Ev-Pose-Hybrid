import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# =================配置区域=================
# 请修改为你解压后的真实路径
# 结构应该是: DATA_ROOT/000000/ev_histogram/*.png
DATA_ROOT = "./ycb_ev_data/dataset/test_pbr" 
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================

class YCBEventDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_list = []

        # 1. 遍历所有物体文件夹 (000000, 000001...)
        obj_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))
        
        print("正在扫描数据集，请稍候...")
        for obj_dir in obj_dirs:
            if not os.path.isdir(obj_dir): continue
            
            # 读取 Ground Truth 标签
            gt_path = os.path.join(obj_dir, "scene_gt.json")
            if not os.path.exists(gt_path): continue
            
            with open(gt_path, 'r') as f:
                scene_gt = json.load(f)
            
            # 遍历该物体下的所有图片
            # JSON 的 key 是 "0", "1", ... 对应图片名 000000.png, 000001.png
            for frame_id_str, gt_data in scene_gt.items():
                # 构造图片文件名: 比如 id "5" -> "000005.png"
                img_name = f"{int(frame_id_str):06d}.png"
                img_path = os.path.join(obj_dir, "ev_histogram", img_name)
                
                # 只有图片存在时才加入列表
                if os.path.exists(img_path):
                    # 提取姿态标签
                    # gt_data[0] 通常包含 'cam_R_m2c' (旋转) 和 'cam_t_m2c' (平移)
                    # 注意：json里可能是一个列表，我们取第一个假设只有一个物体
                    pose_data = gt_data[0] 
                    cam_R = np.array(pose_data['cam_R_m2c']).reshape(3, 3)
                    cam_t = np.array(pose_data['cam_t_m2c'])
                    
                    self.data_list.append({
                        'path': img_path,
                        'R': cam_R,
                        't': cam_t
                    })

        print(f"数据集加载完成，共找到 {len(self.data_list)} 张图片。")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 1. 读取图片
        # ev_histogram 可能是单通道或3通道，统一转为 RGB 方便喂给 ResNet
        image = Image.open(item['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # 2. 处理标签 (6DoF -> 7个数值)
        # 平移 (Translation): 原始单位通常是 mm，我们需要转成 m (除以1000) 方便网络回归
        # 否则 t 的数值太大 (e.g. 500)，而 R 的数值很小 (0-1)，Loss 会爆炸
        t_norm = torch.tensor(item['t'] / 1000.0, dtype=torch.float32)
        
        # 旋转 (Rotation): 3x3 矩阵 -> 四元数 (Quaternion, 4维)
        # 神经网络回归四元数比回归矩阵容易
        quat = R.from_matrix(item['R']).as_quat() # 返回 [x, y, z, w]
        q_norm = torch.tensor(quat, dtype=torch.float32)
        
        # 拼接成 7 维向量 [tx, ty, tz, qx, qy, qz, qw]
        label = torch.cat((t_norm, q_norm), dim=0)
        
        return image, label

def get_resnet_model():
    # 加载预训练的 ResNet18
    model = models.resnet18(pretrained=True)
    
    # 修改全连接层 (FC)
    # 原始 ResNet 输出 1000 类，我们要输出 7 个值
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    
    return model

def main():
    # 1. 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 确保大小一致
        transforms.ToTensor(),
        # ImageNet 的标准化参数 (因为我们用了 pretrained weights)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. 实例化数据集
    full_dataset = YCBEventDataset(DATA_ROOT, transform=transform)
    
    # 防止数据集为空报错
    if len(full_dataset) == 0:
        print("错误：未找到数据！请检查 DATA_ROOT 路径是否正确。")
        return

    # 3. 划分训练集和验证集 (80% / 20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=20)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=20)

    # 4. 模型与优化器
    model = get_resnet_model().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    # 损失函数：分别计算 位置Loss 和 旋转Loss
    # 这里简单地直接用 MSE (均方误差)
    criterion = nn.MSELoss()

    print(f"开始训练... 设备: {DEVICE}")

    # 5. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # 计算 Loss
            # 你可以尝试给旋转加权，例如: loss = loss_t + 10 * loss_q
            # loss = criterion(outputs, labels)

            # 1. 定义两个 Loss
            criterion_t = nn.MSELoss()
            criterion_q = nn.MSELoss() # 或者用 L1Loss，有时候对四元数更好

            # 2. 在训练循环里
            # outputs: [Batch, 7] -> 前3个是t，后4个是q
            pred_t = outputs[:, :3]
            pred_q = outputs[:, 3:]

            label_t = labels[:, :3]
            label_q = labels[:, 3:]

            # 计算分项 Loss
            loss_t = criterion_t(pred_t, label_t)
            loss_q = criterion_q(pred_q, label_q)

            # 组合 Loss (这就是你的超参数，可以写进报告里说你试了 1, 10, 50)
            lambda_rot = 10.0 
            loss = loss_t + lambda_rot * loss_q
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # pbar.set_postfix({'Loss': loss.item()})
            pbar.set_postfix({'L_t': loss_t.item(), 'L_q': loss_q.item()})
            
        epoch_loss = running_loss / len(train_loader)
        
        # 6. 验证循环
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} Summary: Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

         # 【新增】更新学习率
        scheduler.step()
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")

    # 保存模型
    torch.save(model.state_dict(), "resnet18_pose_baseline.pth")
    print("训练完成，模型已保存！")

if __name__ == "__main__":
    main()