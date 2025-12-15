import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
import cv2

# ================= 配置 =================
# 选一张图片来看
IMG_PATH = "./ycb_ev_data/dataset/test_pbr/000000/ev_histogram/000800.png"
MODEL_PATH = "resnet18_pose_baseline.pth" # 用你效果最好的那个模型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =======================================

# 定义简单的 3D 框 (假设物体是个 10cm x 10cm x 10cm 的立方体)
# 实际上 YCB 物体形状各异，但这只是为了演示姿态
def get_cube_points():
    points = []
    r = 0.05 # 半径 5cm
    for x in [-r, r]:
        for y in [-r, r]:
            for z in [-r, r]:
                points.append([x, y, z])
    return np.array(points)

def project_points(points, R_mat, t_vec, K):
    # points: N x 3
    # R: 3 x 3
    # t: 3
    # K: 3 x 3 (相机内参)
    
    # 变换到相机坐标系: P_cam = R * P_obj + t
    points_cam = np.dot(points, R_mat.T) + t_vec
    
    # 投影到像素平面: P_pix = K * P_cam
    points_pix = np.dot(points_cam, K.T)
    
    # 归一化: u = x/z, v = y/z
    u = points_pix[:, 0] / points_pix[:, 2]
    v = points_pix[:, 1] / points_pix[:, 2]
    
    return np.stack([u, v], axis=1)

def main():
    # 1. 加载模型结构 (Baseline ResNet18)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 2. 读取图片
    raw_img = Image.open(IMG_PATH).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)

    # 3. 预测
    with torch.no_grad():
        output = model(input_tensor).cpu().numpy()[0]
    
    # 解析预测值
    pred_t = output[:3] * 1000.0 # 记得乘回 1000 转成 mm (如果训练时除过的话，这里要看你训练时的单位)
    # ⚠️ 注意：如果你训练时除以1000把单位变米了，这里要根据实际情况还原。
    # 假设你 dataset 里是 t_norm = t / 1000.0，那预测出来也是米。
    # 但画图时相机内参通常单位是像素和毫米，或者统一单位。
    # 我们假设 Dataset 里 t 是米，这里 cam_K 也是适配米的。
    
    # 这里有点小坑：YCB 的内参通常是像素单位。t如果是米，需要转换。
    # 简单起见，我们假设 t_norm 就是米。
    pred_t = output[:3] # 米
    pred_q = output[3:] # 四元数
    pred_R = R.from_quat(pred_q).as_matrix()

    # 4. 简易相机内参 (Davids 346 大概参数)
    # fx, fy, cx, cy
    K = np.array([
        [260, 0, 173],
        [0, 260, 130],
        [0, 0, 1]
    ])
    # 注意：如果图片被 resize 成了 224x224，内参也要缩放！
    # 原始大概是 346x260 -> 224x224
    scale_x = 224 / 346
    scale_y = 224 / 260
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] *= scale_x
    K[1, 2] *= scale_y

    # 5. 投影 3D 框
    cube = get_cube_points()
    # 预测框 (Green)
    pts_pred = project_points(cube, pred_R, pred_t, K)

    # 6. 画图
    img_np = np.array(raw_img.resize((224, 224)))
    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)
    
    # 画点
    plt.scatter(pts_pred[:, 0], pts_pred[:, 1], c='lime', s=20, label='Prediction')
    
    plt.legend()
    plt.title("Pose Prediction Visualization")
    plt.savefig("vis_result.png")
    print("可视化结果已保存为 vis_result.png")

if __name__ == "__main__":
    main()