import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler # 新版PyTorch建议写法，旧版用 torch.cuda.amp 也可以
import numpy as np
import os
from tqdm import tqdm
import cv2
import json  # [新增] 用于保存数据
import matplotlib.pyplot as plt

# 引入你的模块
from f2dataset import GMGPoseDataset
from d3model import GMGPVNet
from a5loss import PVNetLoss

## 只训练一种物体
# ================= 1. 全局配置 =================
CONFIG = {

    # --- 实验开关 (修改这里来切换实验) ---
    "exp_name": "all_objects_v1",   # 实验名: 'no_points' 或 'with_points'
    "use_event_points": True,    # 开关: False (不加点云) / True (加点云)
    # -----------------------------------
    "target_obj_id": None,
    
    "processed_dir": "../dataset/processed_data",
    "dataset_root": "../dataset/test_pbr", 
    "batch_size": 32,
    "num_workers": 6,
    "lr": 1e-4,
    "epochs": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "base_save_dir": "./cloudcheckpoint1223",
    "visualize_freq": 1 
}

# 动态生成保存路径，防止覆盖
SAVE_DIR = os.path.join(CONFIG["base_save_dir"], CONFIG["exp_name"])

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


def save_loss_json(history, save_path):
    """[新增] 保存 Loss 数据到 JSON 文件，方便后续对比"""
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=4)

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

# ================= 3. 主训练流程 =================
def train():
    # 初始化当前实验的目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    vis_dir = os.path.join(SAVE_DIR, "vis_logs")
    
    print(f"🚀 Experiment: {CONFIG['exp_name']}")
    print(f"📂 Saving to: {SAVE_DIR}")
    print(f"☁️ Use Point Cloud: {CONFIG['use_event_points']}")

    # 加载数据
    dataset = GMGPoseDataset(
        processed_dir=CONFIG["processed_dir"], 
        dataset_root=CONFIG["dataset_root"],
        target_size=(128, 128),
        mode='train',
        target_obj_id=CONFIG["target_obj_id"] # <--- 传入这里

    )
    
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], 
                        shuffle=True, num_workers=CONFIG["num_workers"],
                        pin_memory=True, drop_last=True)
    
    model = GMGPVNet(num_keypoints=9).to(CONFIG["device"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    criterion = PVNetLoss().to(CONFIG["device"])
    scaler = GradScaler()

    best_loss = float('inf')
    loss_history = {'total': [], 'vec': [], 'seg': []}

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        meter_total = 0.0
        meter_vec = 0.0
        meter_seg = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
        
        for batch in pbar:
            inputs = batch['input'].to(CONFIG["device"])   
            depth = batch['depth'].to(CONFIG["device"])    
            gt_vec = batch['target_field'].to(CONFIG["device"])
            gt_mask = batch['mask'].to(CONFIG["device"])
            # [新增] 获取 Template
            template = batch['template'].to(CONFIG["device"])
            # [核心逻辑] 根据配置决定是否传点云
            if CONFIG["use_event_points"]:
                event_points = batch['event_points'].to(CONFIG["device"])
            else:
                event_points = None # 传 None，模型内部就会跳过 PointNet 分支
            
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                # 传入 event_points (可能是 Tensor 也可能是 None)
                pred_vec, pred_mask = model(inputs, depth,template,event_points=event_points)
                
                _, l_vec, l_seg = criterion(pred_vec, pred_mask, gt_vec, gt_mask)
                # weighted_loss = l_seg + 50.0 * l_vec 
            # === [核心修改] 动态权重策略 ===
            # 阶段 1 (Epoch 1-5): 专注学习 Mask，向量场权重很低或为0
            # 阶段 2 (Epoch 6-50): Mask 稳定了，大力训练向量场
            
            if epoch <= 10:
                w_seg = 1.0
                w_vec = 0.0  # 或者 1.0，先别给太大压力
            else:
                w_seg = 1.0
                # Mask 已经学会了，现在开始暴力拉扯向量场
                # 之前 10.0 不够，现在可以试 20.0 或 50.0 (100可能还是太激进，建议先试 20)
                w_vec = 10.0 
            
            weighted_loss = w_seg * l_seg + w_vec * l_vec 
            # ============================

            scaler.scale(weighted_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            meter_total += weighted_loss.item()
            meter_vec += l_vec.item()
            meter_seg += l_seg.item()
            
            pbar.set_postfix({"Loss": f"{weighted_loss.item():.3f}"})
            scheduler.step()


        # --- Epoch 结束 ---
        avg_total = meter_total / len(loader)
        avg_vec = meter_vec / len(loader)
        avg_seg = meter_seg / len(loader)
        
        # 记录数据
        loss_history['total'].append(avg_total)
        loss_history['vec'].append(avg_vec)
        loss_history['seg'].append(avg_seg)
        
        # 保存日志 (每次都覆盖更新，防止中断丢失)
        save_loss_json(loss_history, f"{SAVE_DIR}/loss_log.json")
        plot_loss_curve(loss_history, f"{SAVE_DIR}/loss_curve.png")
        
        print(f"Ep {epoch} | Total: {avg_total:.4f} (Vec: {avg_vec:.4f})")
        
        # 保存模型
        torch.save(model.state_dict(), f"{SAVE_DIR}/last.pth")
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(model.state_dict(), f"{SAVE_DIR}/best.pth")

        if epoch % CONFIG["visualize_freq"] == 0:
            save_visualization(epoch, batch, pred_vec, pred_mask, vis_dir)

if __name__ == "__main__":
    train()