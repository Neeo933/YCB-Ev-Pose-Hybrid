import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler # 新版PyTorch建议写法，旧版用 torch.cuda.amp 也可以
import numpy as np
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

# 引入你的模块
from a2dataset import GMGPoseDataset
from a3model import GMGPVNet
from a5loss import PVNetLoss

# ================= 1. 全局配置 =================
CONFIG = {
    "processed_dir": "../dataset/processed_data",
    "dataset_root": "../dataset/test_pbr", 
    "batch_size": 32,
    "num_workers": 6,
    "lr": 1e-4,
    "epochs": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "./checkpointsv4",
    "visualize_freq": 1 
}

# ================= 2. 辅助工具：绘制 Loss 曲线 =================
def plot_loss_curve(history, save_path):
    """
    绘制训练过程中的 Loss 变化曲线
    history: {'total': [], 'vec': [], 'seg': []}
    """
    epochs = range(1, len(history['total']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 子图1: Total Loss (加权后的)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['total'], 'b-', label='Weighted Total Loss')
    plt.title('Total Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 子图2: 分项 Loss (原始值)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['vec'], 'r-', label='Vector Loss (Raw)')
    plt.plot(epochs, history['seg'], 'g-', label='Seg Loss (Raw)')
    plt.title('Component Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ================= 3. 可视化监控函数 =================
def save_visualization(epoch, batch_data, pred_vec, pred_mask, save_path):
    inputs = batch_data['input']
    depth = batch_data['depth']
    gt_mask = batch_data['mask']
    gt_vec = batch_data['target_field']
    
    # 1. 还原 RGB
    rgb = inputs[0, :3].cpu().detach().numpy().transpose(1, 2, 0)
    rgb = (rgb * 255).astype(np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # 2. 还原 Depth
    d_vis = depth[0, 0].cpu().detach().numpy()
    
    # 3. Mask 处理
    m_gt = gt_mask[0, 0].cpu().detach().numpy()
    m_pred = torch.sigmoid(pred_mask[0, 0]).cpu().detach().numpy()
    
    # 4. Vector Field (X channel)
    def norm_v(v):
        v = v[0, 0].cpu().detach().numpy()
        return (v - v.min()) / (v.max() - v.min() + 1e-6)
    
    v_gt = norm_v(gt_vec)
    v_pred = norm_v(pred_vec)

    # 5. 绘图
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    
    axs[0,0].imshow(rgb)
    axs[0,0].set_title(f"Ep{epoch} Input RGB")
    
    axs[0,1].imshow(d_vis, cmap='plasma')
    axs[0,1].set_title("Input Depth")
    
    axs[0,2].imshow(m_gt, cmap='gray')
    axs[0,2].set_title("GT Mask") 
    
    axs[0,3].imshow(m_pred, cmap='gray')
    axs[0,3].set_title("Pred Mask Prob")
    
    axs[1,0].imshow(v_gt, cmap='jet')
    axs[1,0].set_title("GT Vector X")
    
    axs[1,1].imshow(v_pred, cmap='jet')
    axs[1,1].set_title("Pred Vector X")
    
    axs[1,2].axis('off'); axs[1,3].axis('off')
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/epoch_{epoch:03d}.png")
    plt.close()

# ================= 4. 主训练流程 =================
def train():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    vis_dir = os.path.join(CONFIG["save_dir"], "vis_logs")
    
    print(f"Loading Data from {CONFIG['dataset_root']}...")
    dataset = GMGPoseDataset(
        processed_dir=CONFIG["processed_dir"], 
        dataset_root=CONFIG["dataset_root"],
        target_size=(128, 128),
        mode='train'
    )
    
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], 
                        shuffle=True, num_workers=CONFIG["num_workers"],
                        pin_memory=True, drop_last=True)
    
    model = GMGPVNet(num_keypoints=9).to(CONFIG["device"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = PVNetLoss().to(CONFIG["device"])
    scaler = GradScaler()

    best_loss = float('inf')
    
    # [新增] 用于记录历史 Loss
    loss_history = {'total': [], 'vec': [], 'seg': []}

    print("🚀 Start Training!")
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        
        # 累计器
        meter_total = 0.0
        meter_vec = 0.0
        meter_seg = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
        
        for batch in pbar:
            inputs = batch['input'].to(CONFIG["device"])   
            depth = batch['depth'].to(CONFIG["device"])    
            gt_vec = batch['target_field'].to(CONFIG["device"])
            
            # [重要提示] 
            # 这里的 gt_mask 应该是你在 dataset.py 里用深度图截断法生成的 mask
            # 如果你 dataset.py 还是旧版，这里 mask 还是方形的。请确保 dataset.py 已更新！
            gt_mask = batch['mask'].to(CONFIG["device"])
            
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                pred_vec, pred_mask = model(inputs, depth, event_points=None)
                
                # l_vec, l_seg 是原始 Loss
                _, l_vec, l_seg = criterion(pred_vec, pred_mask, gt_vec, gt_mask)

                # [核心] 手动加权：向量场权重 = 10.0
                # 这是真正用于反向传播的 loss
                weighted_loss = l_seg + 10.0 * l_vec 

            scaler.scale(weighted_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 记录数据
            meter_total += weighted_loss.item()
            meter_vec += l_vec.item()
            meter_seg += l_seg.item()
            
            pbar.set_postfix({
                "W_Total": f"{weighted_loss.item():.3f}", # 显示加权后的总Loss
                "Vec": f"{l_vec.item():.4f}",
                "Seg": f"{l_seg.item():.3f}"
            })

        # --- Epoch 结束处理 ---
        avg_total = meter_total / len(loader)
        avg_vec = meter_vec / len(loader)
        avg_seg = meter_seg / len(loader)
        
        # 1. 更新历史记录
        loss_history['total'].append(avg_total)
        loss_history['vec'].append(avg_vec)
        loss_history['seg'].append(avg_seg)
        
        # 2. 绘制并保存曲线图
        plot_loss_curve(loss_history, f"{CONFIG['save_dir']}/loss_curve.png")
        
        print(f"Ep {epoch} Done | Weighted Total: {avg_total:.4f} (Vec: {avg_vec:.4f}, Seg: {avg_seg:.4f})")
        
        # 3. 保存模型
        torch.save(model.state_dict(), f"{CONFIG['save_dir']}/last.pth")
        
        # 使用加权后的 Total Loss 来判断最佳模型
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(model.state_dict(), f"{CONFIG['save_dir']}/best.pth")
            print("🏆 New Best Model Saved!")

        # 4. 可视化
        if epoch % CONFIG["visualize_freq"] == 0:
            save_visualization(epoch, batch, pred_vec, pred_mask, vis_dir)

if __name__ == "__main__":
    train()