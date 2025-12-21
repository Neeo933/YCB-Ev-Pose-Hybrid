import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
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
    # 路径设置 (请确保这些路径存在)
    "processed_dir": "./processed_data",
    "dataset_root": "../ycb_ev_data/dataset/test_pbr", 
    
    # 训练超参数
    "batch_size": 16,          # 显存不够改小 (8 或 4)
    "num_workers": 6,          # CPU 核心数
    "lr": 1e-4,                # 初始学习率
    "epochs": 50,              # 训练轮数
    
    # 硬件与保存
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "./checkpoints",
    "visualize_freq": 1        # 每多少个 Epoch 保存一次可视化图片
}


# ================= 2. 可视化监控函数 (升级版) =================
def save_visualization(epoch, batch_data, pred_vec, pred_mask, save_path):
    """
    升级版可视化：增加 Pred Mask 展示，调整布局为 2行4列
    """
    # 取 Batch 中的第一张图
    inputs = batch_data['input']
    depth = batch_data['depth']
    # 注意：这里的 gt_mask 应该是我们用 depth 算出来的那个，稍后在 train 里传入
    # 这里我们先取 dataset 里的原始 mask 做对比
    gt_mask_orig = batch_data['mask']
    gt_vec = batch_data['target_field']
    
    # 1. 还原 RGB
    rgb = inputs[0, :3].cpu().detach().numpy().transpose(1, 2, 0)
    rgb = (rgb * 255).astype(np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # 2. 还原 Depth
    d_vis = depth[0, 0].cpu().detach().numpy()
    
    # 3. Mask 处理
    # 原始方形 Mask
    m_gt_box = gt_mask_orig[0, 0].cpu().detach().numpy()
    # 预测 Mask (Sigmoid -> 0~1)
    m_pred = torch.sigmoid(pred_mask[0, 0]).cpu().detach().numpy()
    
    # 4. Vector Field (归一化显示 X 分量)
    def norm_v(v):
        v = v[0, 0].cpu().detach().numpy()
        return (v - v.min()) / (v.max() - v.min() + 1e-6)
    
    v_gt = norm_v(gt_vec)
    v_pred = norm_v(pred_vec)

    # 5. 绘图 (2行4列)
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    
    # --- Row 1: 输入与 Mask ---
    axs[0,0].imshow(rgb)
    axs[0,0].set_title(f"Ep{epoch} Input RGB")
    
    axs[0,1].imshow(d_vis, cmap='plasma')
    axs[0,1].set_title("Input Depth")
    
    axs[0,2].imshow(m_gt_box, cmap='gray')
    axs[0,2].set_title("GT Mask (Box/Depth)") # 看看是不是变成了轮廓？
    
    axs[0,3].imshow(m_pred, cmap='gray')
    axs[0,3].set_title("Pred Mask Prob") # <--- 这里就是你要看的预测Mask
    
    # --- Row 2: 向量场 ---
    axs[1,0].imshow(v_gt, cmap='jet')
    axs[1,0].set_title("GT Vector X")
    
    axs[1,1].imshow(v_pred, cmap='jet')
    axs[1,1].set_title("Pred Vector X")
    
    # 留两个空位或者画点别的
    axs[1,2].axis('off')
    axs[1,3].axis('off')
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/epoch_{epoch:03d}.png")
    plt.close()

# ================= 3. 主训练流程 =================
def train():
    # 初始化目录
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    vis_dir = os.path.join(CONFIG["save_dir"], "vis_logs1221")
    
    # --- A. 数据加载 ---
    print(f"正在加载数据 (Root: {CONFIG['dataset_root']})...")
    dataset = GMGPoseDataset(
        processed_dir=CONFIG["processed_dir"], 
        dataset_root=CONFIG["dataset_root"],
        target_size=(128, 128),
        mode='train'
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    print(f"加载完成，共 {len(dataset)} 个样本。")

    # --- B. 模型构建 ---
    print("正在构建 GMG-PVNet...")
    model = GMGPVNet(num_keypoints=9).to(CONFIG["device"])
    
    # --- C. 优化器与 Loss ---
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = PVNetLoss().to(CONFIG["device"])
    scaler = GradScaler() # 混合精度训练

    best_loss = float('inf')

    # --- D. 训练循环 ---
    print("🚀 开始训练!")
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        vec_loss_sum = 0.0
        seg_loss_sum = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
        
        for batch in pbar:
            # 1. 数据搬运到 GPU
            inputs = batch['input'].to(CONFIG["device"])   # [B, 4, H, W]
            depth = batch['depth'].to(CONFIG["device"])    # [B, 1, H, W]
            gt_vec = batch['target_field'].to(CONFIG["device"])
            gt_mask = batch['mask'].to(CONFIG["device"])
            
            # 2. 前向传播 (混合精度)
            optimizer.zero_grad()
            with autocast():
                # 注意：这里还没有用 event_points，设为 None
                pred_vec, pred_mask = model(inputs, depth, event_points=None)
                
                # 计算 Loss
                loss, l_vec, l_seg = criterion(pred_vec, pred_mask, gt_vec, gt_mask)

                weighted_loss = l_seg + 10.0 * l_vec 

            
            # 3. 反向传播
            scaler.scale(weighted_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 4. 记录日志
            epoch_loss += loss.item()
            vec_loss_sum += l_vec.item()
            seg_loss_sum += l_seg.item()
            
            pbar.set_postfix({
                "Total": f"{loss.item():.3f}",
                "Vec": f"{l_vec.item():.3f}",
                "Seg": f"{l_seg.item():.3f}"
            })

        # --- E. Epoch 总结 ---
        avg_loss = epoch_loss / len(loader)
        avg_vec = vec_loss_sum / len(loader)
        avg_seg = seg_loss_sum / len(loader)
        
        print(f"Epoch {epoch} 结束 | Total: {avg_loss:.4f} (Vec: {avg_vec:.4f}, Seg: {avg_seg:.4f})")
        
        # 保存模型
        torch.save(model.state_dict(), f"{CONFIG['save_dir']}/last.pth")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{CONFIG['save_dir']}/best.pth")
            print("🏆 新的最佳模型已保存!")

        # 可视化

        if epoch % CONFIG["visualize_freq"] == 0:
            save_visualization(epoch, batch, pred_vec, pred_mask, vis_dir)

if __name__ == "__main__":
    train()