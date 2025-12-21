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
import matplotlib.pyplot as plt

# ================= ⚙️ 配置区域 =================
# 指向 dataset 根目录 (确保里面有 000000/ev_histogram)
DATA_ROOT = "../ycb_ev_data/dataset/test_pbr" 

# 超参数
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 15
LAMBDA_ROT = 20.0      # 旋转Loss的权重 (经验值 10~50)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 结果保存路径
SAVE_DIR = "./results_baseline"
os.makedirs(SAVE_DIR, exist_ok=True)
# ===============================================

class YCBEventDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_list = []

        # 1. 遍历所有物体文件夹
        obj_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))
        
        print("正在扫描数据集 (Baseline)...")
        for obj_dir in obj_dirs:
            if not os.path.isdir(obj_dir): continue
            
            # 读取 Ground Truth 标签
            gt_path = os.path.join(obj_dir, "scene_gt.json")
            if not os.path.exists(gt_path): continue
            
            with open(gt_path, 'r') as f:
                scene_gt = json.load(f)
            
            # 遍历该物体下的所有图片
            for frame_id_str, gt_data in scene_gt.items():
                img_name = f"{int(frame_id_str):06d}.png"
                img_path = os.path.join(obj_dir, "ev_histogram", img_name)
                
                if os.path.exists(img_path):
                    pose_data = gt_data[0] 
                    cam_R = np.array(pose_data['cam_R_m2c']).reshape(3, 3)
                    cam_t = np.array(pose_data['cam_t_m2c'])
                    
                    self.data_list.append({
                        'path': img_path,
                        'R': cam_R,
                        't': cam_t
                    })

        print(f"✅ 数据集加载完成，共找到 {len(self.data_list)} 张图片。")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 读取图片：Baseline 通常使用官方的直方图，转为 RGB 以适配 ResNet 预训练权重
        image = Image.open(item['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # 处理标签
        # 平移: mm -> m
        t_norm = torch.tensor(item['t'] / 1000.0, dtype=torch.float32)
        
        # 旋转: Matrix -> Quaternion
        quat = R.from_matrix(item['R']).as_quat() 
        q_norm = torch.tensor(quat, dtype=torch.float32)
        
        # 拼接 [tx, ty, tz, qx, qy, qz, qw]
        label = torch.cat((t_norm, q_norm), dim=0)
        
        return image, label

def get_resnet_model():
    # 使用 ResNet18 默认权重
    model = models.resnet18(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    return model

def calculate_metrics(pred, target):
    """【新增】计算物理意义上的误差: 厘米(cm) 和 角度(deg)"""
    # pred/target: [B, 7]
    
    # 1. 位置误差 (Euclidean Distance)
    # 输入单位是米，乘以100变厘米
    pos_pred = pred[:, :3]
    pos_target = target[:, :3]
    pos_error_m = torch.norm(pos_pred - pos_target, dim=1)
    pos_error_cm = pos_error_m * 100.0
    
    # 2. 旋转误差 (Geodesic Distance)
    # 角度误差 = 2 * arccos( |<q1, q2>| )
    q_pred = pred[:, 3:]
    q_target = target[:, 3:]
    
    # 归一化四元数 (很重要! 网络输出的四元数模长不一定是1)
    q_pred = torch.nn.functional.normalize(q_pred, dim=1)
    q_target = torch.nn.functional.normalize(q_target, dim=1)
    
    # 点积
    dot_product = torch.abs(torch.sum(q_pred * q_target, dim=1))
    # 防止数值误差导致 arccos 越界
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    angle_error_rad = 2 * torch.acos(dot_product)
    angle_error_deg = torch.rad2deg(angle_error_rad)
    
    return pos_error_cm.mean().item(), angle_error_deg.mean().item()

def plot_history(history):
    """【新增】画 Loss 和 Error 曲线"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # 1. Loss 曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. 位置误差曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['val_pos_err'], 'g-')
    plt.title('Position Error (cm)')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Error (cm)')
    plt.grid(True)

    # 3. 旋转误差曲线
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_rot_err'], 'm-')
    plt.title('Rotation Error (deg)')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Error (degrees)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "baseline_metrics.png"))
    print(f"📊 曲线图已保存至 {SAVE_DIR}/baseline_metrics.png")

def main():
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = YCBEventDataset(DATA_ROOT, transform=transform)
    
    if len(full_dataset) == 0:
        print("错误：未找到数据！")
        return

    # 划分数据集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 注意：如果 num_workers=20 报错，请改为 4 或 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=20)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=20)

    model = get_resnet_model().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # criterion = nn.MSELoss()
     # 定义两个 Loss
    criterion_t = nn.MSELoss()
    criterion_q = nn.L1Loss() # 四元数用 L1 往往更好收敛

    # 记录历史
    history = {'train_loss': [], 'val_loss': [], 'val_pos_err': [], 'val_rot_err': []}
    best_val_loss = float('inf')

    print(f"🚀 开始训练 Baseline 模型 | Device: {DEVICE}")

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            # loss = criterion(outputs, labels)

            # 拆分 Loss
            loss_t = criterion_t(outputs[:, :3], labels[:, :3])
            loss_q = criterion_q(outputs[:, 3:], labels[:, 3:])
            
            # 加权求和
            loss = loss_t + LAMBDA_ROT * loss_q

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        pos_errors = []
        rot_errors = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                
                # Val Loss
                l_t = criterion_t(outputs[:, :3], labels[:, :3])
                l_q = criterion_q(outputs[:, 3:], labels[:, 3:])
                batch_loss = l_t + LAMBDA_ROT * l_q
                val_loss += batch_loss.item()
                
                # 计算物理误差
                p_err, r_err = calculate_metrics(outputs, labels)
                pos_errors.append(p_err)
                rot_errors.append(r_err)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_pos_err = sum(pos_errors) / len(pos_errors)
        avg_rot_err = sum(rot_errors) / len(rot_errors)
        
        # 记录历史
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_pos_err'].append(avg_pos_err)
        history['val_rot_err'].append(avg_rot_err)
        
        # 打印详细日志
        print(f"📝 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   >>> Err Pos: {avg_pos_err:.2f} cm | Err Rot: {avg_rot_err:.2f} deg")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "baseline_model.pth"))
            # print("   💾 Best Model Saved!")

    # 训练结束，画图
    plot_history(history)
    print(f"✨ 训练结束！结果已保存至 {SAVE_DIR}")

if __name__ == "__main__":
    main()