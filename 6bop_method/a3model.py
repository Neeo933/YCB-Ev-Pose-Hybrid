import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# -----------------------------------------------------------
# 基础组件：ResNet Block 提取器
# -----------------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        # 使用 ResNet18 的前几层，去掉 Pooling 以保留空间分辨率
        # 原始 ResNet18 第一层 stride=2, 我们改为 stride=1 保持分辨率
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 加载预训练的 resnet18 (为了简单，这里手写部分层，实际可加载权重)
        # Layer 1: 64 -> 64
        self.layer1 = self._make_layer(64, 64, 2)
        # Layer 2: 64 -> 128 (stride=2, 下采样一次)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        # Layer 3: 128 -> 128 (不再下采样，保持特征丰富)
        self.layer3 = self._make_layer(128, out_channels, 2, stride=1)
        
    def _make_layer(self, in_planes, planes, blocks, stride=1):
        # [核心修正] 如果步长不为1或通道数改变，必须定义 downsample 层来调整 shortcut
        downsample = None
        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        # 将 downsample 传入 BasicBlock
        layers.append(models.resnet.BasicBlock(in_planes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(models.resnet.BasicBlock(planes, planes))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        _, out = self.forward_features(x)
        return out

    def forward_features(self, x):
        # [新增] 返回中间特征用于 Skip Connection
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        l1 = self.layer1(x) # [B, 64, 128, 128] (浅层特征)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2) # [B, 128, 64, 64] (深层特征)
        
        return l1, l3

# -----------------------------------------------------------
# 1. Gated Fusion 模块
# -----------------------------------------------------------
class GatedFusion(nn.Module):
    def __init__(self, channels):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels), # 加上BN训练更稳
            nn.Sigmoid()
        )

    def forward(self, feat_rgb, feat_mts):
        combined = torch.cat([feat_rgb, feat_mts], dim=1)
        alpha = self.gate(combined)
        # 融合: RGB为主，MTS为辅，或者自适应
        fused_feat = alpha * feat_rgb + (1 - alpha) * feat_mts
        return fused_feat

# -----------------------------------------------------------
# 分支 B: Mini PointNet (处理事件点云)
# -----------------------------------------------------------
class PointNetBackbone(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()
        # 简单的 PointNet 实现: (x,y,t,p) -> Feature
        self.mlp1 = nn.Sequential(
            nn.Conv1d(4, 64, 1), # Input: 4 channels (x,y,t,p)
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, N, 4] -> [B, 4, N]
        x = x.transpose(2, 1)
        x = self.mlp1(x)
        x = self.mlp2(x)
        point_features = self.mlp3(x) # [B, 128, N] 每个点的局部特征
        return point_features

# -----------------------------------------------------------
# 2. AFDM 模块：特征扩散
# -----------------------------------------------------------
class AFDM(nn.Module):
    def __init__(self, feat_dim):
        super(AFDM, self).__init__()
        self.feat_dim = feat_dim
        # 用于扩散特征的小卷积核 (高斯模糊的作用)
        self.diffuse_conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU()
        )

    def forward(self, feat_2d, point_coords, point_feats):
        """
        feat_2d: [B, C, H, W]
        point_coords: [B, N, 2] (u, v) 像素坐标
        point_feats: [B, C, N] 点云特征
        """
        B, C, H, W = feat_2d.shape
        device = feat_2d.device
        
        # 1. 初始化空白画布
        # 为了效率，我们可能在较小的分辨率上投影，然后上采样。这里假设直接在特征图分辨率上投影。
        scatter_map = torch.zeros((B, C, H, W), device=device)
        
        # 2. 坐标归一化与离散化
        # 假设 point_coords 是在 128x128 尺度下的绝对坐标
        # 我们需要将其映射到特征图尺寸 (例如 64x64)
        scale_x = W / 128.0 
        scale_y = H / 128.0
        
        # 转为整数索引 (LongTensor)
        idx_x = (point_coords[:, :, 0] * scale_x).long().clamp(0, W-1)
        idx_y = (point_coords[:, :, 1] * scale_y).long().clamp(0, H-1)
        
        # 3. Scatter Add (将点云特征加到画布像素上)
        # 计算扁平化索引: idx = y * W + x
        flat_indices = idx_y * W + idx_x # [B, N]
        
        # PyTorch 的 scatter 需要 batch 维处理，比较繁琐。
        # 这里使用一种循环处理 Batch 的方式 (如果 N 很大，建议用 scatter_nd 或 extension)
        for b in range(B):
            # 获取当前 batch 的特征 [C, N] 和索引 [N]
            p_feat_b = point_feats[b] 
            idx_b = flat_indices[b]
            
            # 展平画布空间维度 [C, H*W]
            flat_map_b = scatter_map[b].view(C, -1)
            
            # 核心：将点特征累加到像素上
            flat_map_b.index_add_(1, idx_b, p_feat_b)
            
            # 恢复形状
            scatter_map[b] = flat_map_b.view(C, H, W)
            
        # 4. 特征扩散 (Diffusion)
        # 因为 scatter 后的图非常稀疏（只有有点的地方有值），需要卷积把它“晕染”开
        diffused_feat = self.diffuse_conv(scatter_map)
        
        # 5. 融合 (残差连接)
        refined_feat = feat_2d + diffused_feat
        return refined_feat

# -----------------------------------------------------------
# 3. GMG-PVNet 主体网络
# -----------------------------------------------------------
class GMGPVNet(nn.Module):
    def __init__(self, num_keypoints=9):
        super(GMGPVNet, self).__init__()
        self.feat_dim = 128
        
        # 分支 A: 2D CNN (RGB)
        self.backbone_rgb = FeatureExtractor(in_channels=3, out_channels=self.feat_dim)
        
        # 分支 B: 几何特征 (MTS + Depth = 2通道)
        self.backbone_geo = FeatureExtractor(in_channels=2, out_channels=self.feat_dim)
        
        # 分支 C: PointNet (可选)
        self.pointnet = PointNetBackbone(out_channels=self.feat_dim)

        # 融合层 (深层)
        self.fusion = GatedFusion(channels=self.feat_dim)
        self.afdm = AFDM(feat_dim=self.feat_dim)

        # [修改点 1] Skip Connection 处理层
        # 因为我们要拼接 RGB(64) 和 Geo(64) 的浅层特征，所以输入是 128
        self.skip_layer = nn.Conv2d(128, 64, kernel_size=1)

        # 上采样模块 (64x64 -> 128x128)
        self.upsample_block = nn.Sequential(
            nn.ConvTranspose2d(self.feat_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 最终聚合层 (Upsampled 64 + Skip 64 = 128)
        self.head_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 预测头
        self.vector_head = nn.Conv2d(64, num_keypoints * 2, kernel_size=1)
        self.seg_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, input_tensor, depth_tensor, event_points=None):
        """
        input_tensor: [B, 4, 128, 128] (RGB+MTS)
        depth_tensor: [B, 1, 128, 128]
        """
        # 1. 拆分输入
        rgb_crop = input_tensor[:, :3, :, :] # [B, 3, 128, 128]
        mts_crop = input_tensor[:, 3:, :, :] # [B, 1, 128, 128]
        
        # 构造几何输入 (MTS + Depth)
        geo_input = torch.cat([mts_crop, depth_tensor], dim=1) # [B, 2, 128, 128]
        
        # 2. Backbone Forward (返回 浅层 和 深层 特征)
        # l1: [B, 64, 128, 128], l3: [B, 128, 64, 64]
        f_rgb_l1, f_rgb_l3 = self.backbone_rgb.forward_features(rgb_crop)
        f_geo_l1, f_geo_l3 = self.backbone_geo.forward_features(geo_input)
        
        # 3. Gated Fusion (在深层特征上融合)
        f_2d = self.fusion(f_rgb_l3, f_geo_l3)
        
        # 4. AFDM 特征扩散 (如果有点云)
        if event_points is not None:
            p_coords = event_points[:, :, :2] 
           # B. 准备 PointNet 需要的输入 (归一化 x,y 到 0-1，与 t,p 匹配)
            # [修改点] 这里非常重要！
            points_norm = event_points.clone()
            points_norm[:, :, 0] = points_norm[:, :, 0] / 128.0 # Normalize X
            points_norm[:, :, 1] = points_norm[:, :, 1] / 128.0 # Normalize Y
            
            # C. 提取点云特征
            p_feats = self.pointnet(points_norm) # [B, 128, N]
            
            # D. 扩散并融合
            f_2d = self.afdm(f_2d, p_coords, p_feats)
        
        # 5. 上采样 (64x64 -> 128x128)
        f_up = self.upsample_block(f_2d) # [B, 64, 128, 128]
        
        # 6. Skip Connection 融合
        # [修改点 2] 拼接 RGB 和 Geo 的浅层特征 (64+64=128) -> skip_layer -> 64
        # 这样既保留了RGB的纹理细节，也保留了边缘几何细节
        f_shallow_cat = torch.cat([f_rgb_l1, f_geo_l1], dim=1) 
        f_skip = self.skip_layer(f_shallow_cat)
        
        # 再次拼接: 上采样后的深层特征(64) + 处理后的浅层特征(64) = 128
        f_final = torch.cat([f_up, f_skip], dim=1) # [B, 128, 128, 128]
        
        f_final = self.head_conv(f_final)
        
        # 7. 输出
        pred_vectors = self.vector_head(f_final)
        pred_mask = self.seg_head(f_final)
        
        return pred_vectors, pred_mask