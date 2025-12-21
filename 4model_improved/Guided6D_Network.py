import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 引导融合模块 (Guided-Fusion Module) ---
class GuidedFusionModule(nn.Module):
    def __init__(self, in_channels):
        super(GuidedFusionModule, self).__init__()
        # 空间注意力：识别 MTS 中的显著几何特征
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        # 特征对齐与微调
        self.refine = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x_depth, x_mts):
        # 计算注意力权重
        attn = self.spatial_attn(x_mts)
        # 引导增强：加强深度特征中属于边缘的部分
        x_depth_refined = x_depth * (1 + attn)
        # 级联融合并对齐语义
        combined = torch.cat([x_depth_refined, x_mts], dim=1)
        out = F.relu(self.norm(self.refine(combined)))
        return out, attn

# --- 2. 密集对应关系预测头 (Pose Header) ---
class PoseHead(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super(PoseHead, self).__init__()
        # 预测物体表面的 3D 坐标 (X, Y, Z)
        self.xyz_regressor = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1) 
        )
        # 同时预测物体掩码 (Mask) 用于排除背景
        self.mask_predictor = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        xyz_map = self.xyz_regressor(x)
        mask_logit = self.mask_predictor(x)
        return xyz_map, mask_logit

# --- 3. 完整网络架构 (Guided6D-Net) ---
class Guided6DNet(nn.Module):
    def __init__(self):
        super(Guided6DNet, self).__init__()
        
        # 采用轻量化 Backbone (例如 ResNet 的基础块)
        # 这里为了演示，使用简化的下采样模块
        def basic_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # 深度流特征提取器 (处理深度归一化后的 Blue 通道)
        self.depth_stream = nn.Sequential(
            basic_block(1, 64),  # H/2
            basic_block(64, 128), # H/4
            basic_block(128, 256) # H/8
        )

        # MTS流特征提取器 (处理 MTS 锐化边缘的 Red 通道)
        self.mts_stream = nn.Sequential(
            basic_block(1, 64),
            basic_block(64, 128),
            basic_block(128, 256)
        )

        # 核心引导融合头
        self.fusion = GuidedFusionModule(256)
        
        # GDR-Net 风格预测头
        self.pose_head = PoseHead(256)

    def forward(self, input_tensor):
        """
        input_tensor: 之前生成的 3-Channel Tensor (B, 3, H, W)
        R: MTS 边缘, G: 深度梯度, B: 原始深度
        """
        # 拆分通道
        mts_chan = input_tensor[:, 0:1, :, :] # 取 R
        depth_chan = input_tensor[:, 2:3, :, :] # 取 B

        # 提取双流特征
        f_depth = self.depth_stream(depth_chan)
        f_mts = self.mts_stream(mts_chan)

        # 引导融合
        f_fused, attn_map = self.fusion(f_depth, f_mts)

        # 预测 XYZ Map 和 Mask
        xyz_map, mask_logit = self.pose_head(f_fused)

        return {
            "xyz_map": xyz_map,      # 目标 3D 坐标回归
            "mask_logit": mask_logit, # 前景分割
            "attention": attn_map     # 供可视化分析
        }

# --- 测试模型实例化 ---
if __name__ == "__main__":
    # 模拟输入 (Batch, Channel, H, W)
    # 假设输入是处理好的 256x256 对齐张量
    test_input = torch.randn(1, 3, 256, 256)
    model = Guided6DNet()
    output = model(test_input)

    print("模型输出维度:")
    print(f"XYZ Map: {output['xyz_map'].shape}")    # 应为 [1, 3, 32, 32] (H/8)
    print(f"Mask Logit: {output['mask_logit'].shape}")
    print(f"Attention Map: {output['attention'].shape}")