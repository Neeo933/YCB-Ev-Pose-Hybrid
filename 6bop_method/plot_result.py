import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(log_path, save_dir):
    """
    读取 loss_log.json 并生成高质量图表
    """
    if not os.path.exists(log_path):
        print(f"❌ 找不到日志文件: {log_path}")
        return

    with open(log_path, 'r') as f:
        data = json.load(f)

    epochs = range(1, len(data['total']) + 1)
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-paper') 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 左图：加权总 Loss (Weighted Total Loss) ---
    ax1.plot(epochs, data['total'], color='#1f77b4', linewidth=2, label='Total Loss')
    ax1.set_title('Training Convergence (Weighted Total Loss)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # --- 右图：分项 Loss 对比 (Vector vs Seg) ---
    # Vector Loss 通常比 Seg Loss 大，建议使用双 Y 轴或者归一化对比
    lns1 = ax2.plot(epochs, data['vec'], color='#d62728', linewidth=1.5, label='Vector Field Loss')
    ax2_twin = ax2.twinx()
    lns2 = ax2_twin.plot(epochs, data['seg'], color='#2ca02c', linewidth=1.5, label='Segmentation Loss')
    
    ax2.set_title('Component Loss Analysis', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Vector Loss', fontsize=12, color='#d62728')
    ax2_twin.set_ylabel('Segmentation Loss', fontsize=12, color='#2ca02c')
    
    # 合并图例
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='upper right')

    plt.tight_layout()
    
    # 自动保存
    save_path = os.path.join(save_dir, "refined_training_report.png")
    plt.savefig(save_path, dpi=300) # 高清保存
    print(f"✅ 结果图已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    # 修改这里指向你的实验目录
    EXP_DIR = "./finalexperiment/exp/20/obj_20_full/"
    LOG_FILE = os.path.join(EXP_DIR, "loss_log.json")
    
    plot_training_results(LOG_FILE, EXP_DIR)