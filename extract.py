import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, convolve
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


def imread_unicode(image_path, flags=cv2.IMREAD_GRAYSCALE):
    """兼容 Windows 中文路径的图像读取。"""
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        if data.size > 0:
            image = cv2.imdecode(data, flags)
            if image is not None:
                return image
    except Exception:
        pass

    try:
        return cv2.imread(image_path, flags)
    except Exception:
        return None


# ============================================================
# Sobel边界检测
# ============================================================

def sobel_boundary_detection(gray_image):
    """使用Sobel算子检测图像边界"""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    Gx = convolve(gray_image.astype(np.float64), sobel_x, mode='nearest')
    Gy = convolve(gray_image.astype(np.float64), sobel_y, mode='nearest')

    gradient_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    threshold = 0.01 * gradient_magnitude.max()
    boundary_image = gradient_magnitude > threshold
    return boundary_image


# ============================================================
# 调制区域分层编号
# ============================================================

def get_modulation_region(original_image, boundary_image, half_bandwidth_modulation=20):
    """获取调制区域并进行分层编号"""
    rows, cols = original_image.shape
    region_map = np.zeros((rows, cols), dtype=np.int32)

    boundary_y, boundary_x = np.where(boundary_image)
    if len(boundary_y) == 0:
        raise ValueError('未检测到边界')

    dist_map = distance_transform_edt(~boundary_image.astype(bool))
    num_layers = half_bandwidth_modulation * 2 + 1
    layer_info = []

    for layer in range(1, num_layers + 1):
        if layer <= half_bandwidth_modulation:
            inner_dist = half_bandwidth_modulation + 1 - layer
            layer_mask = ((dist_map >= inner_dist) & (dist_map < inner_dist + 1)
                          & original_image.astype(bool))
        else:
            outer_dist = layer - half_bandwidth_modulation - 1
            layer_mask = ((dist_map >= outer_dist) & (dist_map < outer_dist + 1)
                          & ~original_image.astype(bool))
            if outer_dist == 0:
                layer_mask = boundary_image.astype(bool)

        region_map[layer_mask] = layer
        layer_y, layer_x = np.where(layer_mask)
        layer_coords = np.column_stack((layer_x, layer_y))
        pixel_vals = original_image[layer_mask]

        layer_info.append({
            'layerNumber': layer,
            'pixelCount': int(np.sum(layer_mask)),
            'pixelCoordinates': layer_coords,
            'pixelValues': pixel_vals,
        })
        print(f'第{layer}层: {int(np.sum(layer_mask))}个像素')

    return region_map, layer_info


# ============================================================
# 显示结果
# ============================================================

def display_results(original_image, boundary_image, region_map, layer_info):
    num_layers = len(layer_info)
    plt.figure('边界与分层结果', figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('输入原图')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(boundary_image, cmap='gray')
    plt.title('Sobel边界')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(region_map, cmap='jet', vmin=0, vmax=num_layers)
    plt.title(f'调制区域分层 (共{num_layers}层)')
    plt.axis('image')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


