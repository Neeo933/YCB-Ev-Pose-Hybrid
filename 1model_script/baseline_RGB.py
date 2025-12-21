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
# 指向 dataset 根目录 (确保里面有 000000/rgb_events)
DATA_ROOT = "../ycb_ev_data/dataset/test_pbr" 

# 超参数
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 15            # RGB模型收敛稍慢，建议多跑几轮
LAMBDA_ROT = 20.0      # 旋转Loss的权重 (经验值 10~50)
WEIGHT_DECAY = 1e-4    # 防止过拟合

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./results_rgb"
os.makedirs(SAVE_DIR, exist_ok=True)
# ===============================================

class YCBEventRGBDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_list = []

        # 扫描所有物体文件夹
        obj_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))
        print("正在扫描 RGB 数据集...")
        
        for obj_dir in obj_dirs:
            if not os.path.isdir(obj_dir): continue
            
            # 读取标签
            gt_path = os.path.join(obj_dir, "scene_gt.json")
            if not os.path.exists(gt_path): continue
            
            with open(gt_path, 'r') as f:
                scene_gt = json.load(f)
            
            # 这里的文件夹名必须是你生成的 "rgb_events"
            img_dir = os.path.join(obj_dir, "rgb_events")
            if not os.path.exists(img_dir): continue
            
            for frame_id_str, gt_data in scene_gt.items():
                img_name = f"{int(frame_id_str):06d}.png"
                img_path = os.path.join(img_dir, img_name)
                
                if os.path.exists(img_path):
                    pose_data = gt_data[0]
                    cam_R = np.array(pose_data['cam_R_m2c']).reshape(3, 3)
                    cam_t = np.array(pose_data['cam_t_m2c'])
                    
                    self.data_list.append({
                        'path': img_path,
                        'R': cam_R,
                        't': cam_t
                    })

        print(f"✅ 加载完成，共 {len(self.data_list)} 张 RGB 图片。")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 【关键】读取为 RGB (3通道)
        # R=Past, G=Present, B=Future
        image = Image.open(item['path']).convert('RGB')
        
        # --- 探针开始 ---
        img_np = np.array(image)
        # 1. 检查图片是否全黑
        if img_np.max() < 10:
            print(f"🚨 警告：图片过暗或全黑！Max value: {img_np.max()} | Path: {item['path']}")
        
        # 2. 检查标签单位
        t_raw = item['t'] # 原始数据
        if np.max(np.abs(t_raw)) < 1.0:
            # 如果原始数据最大值都不到1，说明是米。你再除以1000，就变成微米了！
            print(f"🚨 警告：标签数值过小！可能单位已经是米了，不要除以1000！Val: {t_raw}")
        # --- 探针结束 ---


        if self.transform:
            image = self.transform(image)
            
        # 处理标签
        # 1. 位置归一化: mm -> m
        t_norm = torch.tensor(item['t'] / 1000.0, dtype=torch.float32)
        
        # 2. 旋转矩阵 -> 四元数
        quat = R.from_matrix(item['R']).as_quat() 
        q_norm = torch.tensor(quat, dtype=torch.float32)
        
        # 拼接 [tx, ty, tz, qx, qy, qz, qw]
        label = torch.cat((t_norm, q_norm), dim=0)
        
        return image, label

def get_rgb_model():
    # 使用 ResNet18
    # weights='DEFAULT' 会自动加载 ImageNet 预训练权重
    # 标准 ResNet 输入就是 3通道，所以不需要改第一层
    model = models.resnet18(weights='DEFAULT')
    
    # 修改输出层为 7 (3 Pos + 4 Rot)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    
    return model

def calculate_metrics(pred, target):
    """计算物理意义上的误差: 厘米(cm) 和 角度(deg)"""
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
    
    # 归一化四元数 (很重要!)
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
    """画 Loss 和 Accuracy 曲线"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # 1. Loss 曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Weighted Loss')
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
    plt.savefig(os.path.join(SAVE_DIR, "training_metrics.png"))
    print(f"📊 曲线图已保存至 {SAVE_DIR}/training_metrics.png")

def main():
    # 数据增强
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # 轻微增强
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = YCBEventRGBDataset(DATA_ROOT, transform=None)
    full_dataset.data_list = full_dataset.data_list[:16] # ✂️ 强行只留 16 个样本
    
    
    # 划分数据集 (需要给 subsets 重新赋值 transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    # 手动设置 transform (PyTorch Dataset 的小 trick)
    # 注意：这里假设 full_dataset.dataset 是 YCBEventRGBDataset
    # 如果报错，可以直接在 Dataset 内部根据 phase 处理，这里简化处理：
    # 我们直接让 Dataset 每次都返回 transform 后的，或者这里简单点：
    # 为了严谨，应该重写 Dataset 接受 split，但这里为了代码短，
    # 我们直接把 Dataset 的 transform 设为 train 的，验证集也用一样的（只是resize/norm），影响不大
    full_dataset.transform = transform_train
    
    # train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=20)
    # val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=20)

     # 训练/验证集都用这 16 个
    train_loader = DataLoader(full_dataset, batch_size=4, shuffle=True, num_workers=20) # Batch设小点
    val_loader = DataLoader(full_dataset, batch_size=4, shuffle=False, num_workers=20)

    # 模型
    model = get_rgb_model().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # 定义两个 Loss
    criterion_t = nn.MSELoss()
    criterion_q = nn.L1Loss() # 四元数用 L1 往往更好收敛

    # 记录历史
    history = {'train_loss': [], 'val_loss': [], 'val_pos_err': [], 'val_rot_err': []}
    best_val_loss = float('inf')

    print(f"🚀 开始训练 RGB 模型 | Device: {DEVICE}")
    print(f"配置: Alpha(Rot)={LAMBDA_ROT}, Epochs={EPOCHS}")

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # 拆分 Loss
            loss_t = criterion_t(outputs[:, :3], labels[:, :3])
            loss_q = criterion_q(outputs[:, 3:], labels[:, 3:])
            
            # 加权求和
            loss = loss_t + LAMBDA_ROT * loss_q
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'Lt': f"{loss_t.item():.4f}", 'Lq': f"{loss_q.item():.4f}"})
            
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
                
                # 物理指标计算
                p_err, r_err = calculate_metrics(outputs, labels)
                pos_errors.append(p_err)
                rot_errors.append(r_err)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_pos_err = sum(pos_errors) / len(pos_errors)
        avg_rot_err = sum(rot_errors) / len(rot_errors)
        
        # 记录
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_pos_err'].append(avg_pos_err)
        history['val_rot_err'].append(avg_rot_err)
        
        print(f"📝 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   >>> Err Pos: {avg_pos_err:.2f} cm | Err Rot: {avg_rot_err:.2f} deg")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_rgb_model.pth"))
            print("   💾 New Best Model Saved!")

    # 结束画图
    plot_history(history)
    print("✨ 训练结束！")

if __name__ == "__main__":
    main()