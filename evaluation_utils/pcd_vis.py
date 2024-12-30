import numpy as np
import open3d as o3d

def create_point_cloud_from_rgb_depth(rgb_image, depth_image,intrinsics ,save_name, depth_scale=1.0, max_depth=80.0):
    """
    将 RGB 图像和深度图转换为点云。
    
    参数:
    - rgb_image: RGB 图像，形状为 (H, W, 3)
    - depth_image: 深度图，形状为 (H, W)
    - fx, fy: 相机内参，焦距
    - cx, cy: 相机内参，光心坐标
    - depth_scale: 深度比例因子，将深度值从像素单位转换为米
    - max_depth: 深度值的最大阈值（单位：米）
    
    返回:
    - o3d.geometry.PointCloud 对象
    """
    # 图像高宽
    h, w = depth_image.shape
    
    # 创建网格网点
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # 将深度图转换为实际深度值
    depth = depth_image.astype(np.float32) / depth_scale
    
    # 深度阈值过滤
    valid_mask = (depth > 0) & (depth < max_depth)
    x = x[valid_mask]
    y = y[valid_mask]
    depth = depth[valid_mask]
    
    fx, fy, cx, cy = intrinsics
    # 计算 3D 坐标
    z = depth
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    
    # 构建点云的 XYZ 坐标
    points = np.stack((x, y, z), axis=-1)
    
    # 获取有效的 RGB 值
    colors = rgb_image[valid_mask] / 255.0  # 归一化为 [0, 1]
    
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(save_name, pcd)
    return pcd


import matplotlib
import numpy as np
import torch

def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    # print(img_colored_np.shape)
    # img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored

# 示例：加载 RGB 和深度图像，生成点云并保存
if __name__ == "__main__":
    # 加载 RGB 图像和深度图
    rgb_image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)  # 这里替换为实际加载的图像
    depth_image = np.random.uniform(0, 8000, (400, 600)).astype(np.uint16)  # 假设深度以毫米为单位
    
    # 相机内参（替换为实际相机参数）
    fx, fy = 500.0, 500.0  # 焦距
    cx, cy = 300.0, 200.0  # 光心坐标
    depth_scale = 1000.0   # 深度图的单位是毫米，因此需要除以 1000 转为米
    
    # 生成点云
    pcd = create_point_cloud_from_rgb_depth(rgb_image, depth_image, fx, fy, cx, cy, depth_scale, max_depth=80.0)
    
    # 保存点云到 PCD 文件
    o3d.io.write_point_cloud("output.pcd", pcd)