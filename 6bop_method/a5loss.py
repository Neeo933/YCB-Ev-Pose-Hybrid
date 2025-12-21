import torch
import torch.nn as nn

class PVNetLoss(nn.Module):
    def __init__(self):
        super(PVNetLoss, self).__init__()
        # 向量回归使用 SmoothL1，对离群点不敏感，训练更稳
        self.vector_loss = nn.SmoothL1Loss(reduction='none')
        # 分割使用带 Logits 的二元交叉熵，内部集成了 Sigmoid，数值更稳定
        self.seg_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_vectors, pred_mask, gt_vectors, gt_mask):
        """
        pred_vectors: [B, 18, 128, 128]
        pred_mask:    [B, 1, 128, 128]
        gt_vectors:   [B, 18, 128, 128] (来自 Dataset 的 'vector_field')
        gt_mask:      [B, 1, 128, 128] (来自 Dataset 的 'mask')
        """
        # 1. 计算分割损失 (全图计算)
        loss_seg = self.seg_loss(pred_mask, gt_mask.float())

        # 2. 计算向量场损失 (辩证重点：只在物体内部计算)
        # gt_mask 为 1 的地方才是我们需要预测向量的地方
        mask_active = gt_mask.expand_as(pred_vectors) > 0.5        
        if mask_active.any():
            # 只提取有效像素点
            diff = self.vector_loss(pred_vectors, gt_vectors)
            # 将物体外的损失强制设为 0
            loss_vec = diff[mask_active].mean()
        else:
            # 如果这一帧里没物体（理论上不会，但增加鲁棒性）
            loss_vec = torch.tensor(0.0).to(pred_vectors.device)

        # 3. 复合损失 (可以根据需要调整权重，通常 1:1)
        total_loss = loss_seg + loss_vec
        
        return total_loss, loss_vec, loss_seg

    