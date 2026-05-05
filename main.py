# -*- coding: utf-8 -*-
import math
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.path import Path
import tkinter as tk
from tkinter import filedialog
from extract import imread_unicode, sobel_boundary_detection, get_modulation_region, display_results


def polygon_area(poly):

    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area * 0.5


def extract_boundary_poly(boundary_image, min_points=4, epsilon=None):
    """
    从边界图像中提取边界多边形。

    参数：
    boundary_image (numpy.ndarray): 二值边界图像。
    min_points (int): 最小顶点数,默认4。
    epsilon (float): approxPolyDP 的 epsilon 参数,默认自动计算。

    返回：
    list of tuple: 多边形顶点列表，每个顶点为 (x, y) 元组；如果失败返回 None。
    """
    binary = (boundary_image.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        pts = np.column_stack(np.where(boundary_image.astype(np.uint8) > 0))
        if len(pts) < 3:
            return None
        hull = cv2.convexHull(pts.astype(np.float32))
        if len(hull) < 3:
            return None
        poly = [tuple(pt[0].astype(np.float64)) for pt in hull]
    else:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 10:
            pts = np.column_stack(np.where(boundary_image.astype(np.uint8) > 0))
            if len(pts) < 3:
                return None
            hull = cv2.convexHull(pts.astype(np.float32))
            if len(hull) < 3:
                return None
            poly = [tuple(pt[0].astype(np.float64)) for pt in hull]
        else:
            if epsilon is None:
                peri = cv2.arcLength(contour, True)
                epsilon = max(1.0, 0.005 * peri)

            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx = approx.squeeze()
            if approx.ndim != 2 or len(approx) < 3:
                pts = np.column_stack(np.where(boundary_image.astype(np.uint8) > 0))
                if len(pts) < 3:
                    return None
                hull = cv2.convexHull(pts.astype(np.float32))
                if len(hull) < 3:
                    return None
                poly = [tuple(pt[0].astype(np.float64)) for pt in hull]
            else:
                poly = [tuple(pt.astype(np.float64)) for pt in approx]

    if polygon_area(poly) < 0:
        poly.reverse()
    return poly


def sample_contour(poly, step=1.0):
    """
    在多边形轮廓上采样点。

    参数：
    poly (list of tuple): 多边形顶点列表。
    step (float): 采样步长,默认1.0。

    返回：
    list of tuple: 采样点列表。
    """
    points = []
    for i in range(len(poly)):
        p0 = np.array(poly[i], dtype=np.float64)
        p1 = np.array(poly[(i + 1) % len(poly)], dtype=np.float64)
        dist = np.linalg.norm(p1 - p0)
        if dist == 0:
            continue
        count = max(int(dist // step), 1)
        for t in np.linspace(0, 1, count, endpoint=False):
            points.append(tuple(p0 + t * (p1 - p0)))
    return points


def circle_offsets(radius, spacing=0.8):
    """
    生成圆形区域内的点偏移。

    参数：
    radius (float): 圆半径。
    spacing (float): 点间距,默认0.8。

    返回：
    numpy.ndarray: 偏移点数组，形状 (N, 2)。
    """
    coords = np.arange(-radius, radius + spacing, spacing)
    xx, yy = np.meshgrid(coords, coords)
    mask = xx * xx + yy * yy <= radius * radius
    return np.stack((xx[mask], yy[mask]), axis=1)


def point_in_poly_array(points, poly):
    """
    判断点是否在多边形内。

    参数：
    points (numpy.ndarray): 点数组，形状 (N, 2)。
    poly (list of tuple): 多边形顶点列表。

    返回：
    numpy.ndarray: 布尔数组，表示每个点是否在多边形内。
    """
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0,), dtype=bool)

    path = Path(np.asarray(poly, dtype=np.float64))
    return path.contains_points(points)


def boundary_normal(p_prev, p_next):
    """
    计算边界边的法向量。

    参数：
    p_prev (tuple): 前一个点。
    p_next (tuple): 下一个点。

    返回：
    numpy.ndarray: 法向量。
    """
    edge = np.array(p_next, dtype=np.float64) - np.array(p_prev, dtype=np.float64)
    normal = np.array([edge[1], -edge[0]], dtype=np.float64)
    norm = np.linalg.norm(normal)
    return normal / norm if norm != 0 else np.zeros(2, dtype=np.float64)


def expand_boundary(poly, radius=6.0, step=1.0, arc_segments=12, overlap_spacing=0.8):
    """
    扩展边界，生成遍历路径。

    参数：
    poly (list of tuple): 边界多边形。
    radius (float): 扩展半径,默认6.0。
    step (float): 采样步长,默认1.0。
    arc_segments (int): 弧段数,默认12。
    overlap_spacing (float): 重叠间距,默认0.8。

    返回：
    list of tuple: 扩展后的点列表。
    """
    samples = sample_contour(poly, step)
    offsets = circle_offsets(radius, spacing=overlap_spacing)
    def overlap_ratio(center):
        pts = offsets + np.array(center, dtype=np.float64)
        return point_in_poly_array(pts, poly).mean()

    d_values = [((0.5 - overlap_ratio(pt)) / 0.25) ** 2 if overlap_ratio(pt) < 0.5 else 0.0
                for pt in samples]

    vertex_indices = {}
    for i, pt in enumerate(samples):
        for j, v in enumerate(poly):
            if np.linalg.norm(np.array(pt) - np.array(v, dtype=np.float64)) < 1e-8:
                vertex_indices[i] = j
                break

    expanded = []
    for i, pt in enumerate(samples):
        if i == 0:
            normal = boundary_normal(samples[-1], samples[(i + 1) % len(samples)])
            expanded.append(tuple(np.array(pt, dtype=np.float64) + d_values[i] * normal))
        else:
            expanded.append(pt)


def show_expansion(original_image, boundary_image, region_map, layer_info, boundary_poly, expanded_poly):

    display_results(original_image, boundary_image, region_map, layer_info)
    plt.figure('边界遍历结果', figsize=(8, 8))
    poly = np.array(boundary_poly + [boundary_poly[0]])
    exp = np.array(expanded_poly + [expanded_poly[0]])
    plt.plot(poly[:, 0], poly[:, 1], 'b-', linewidth=2, label='识别边界')
    plt.plot(exp[:, 0], exp[:, 1], 'r--', linewidth=2, label='遍历结果')
    plt.fill(poly[:, 0], poly[:, 1], color='lightblue', alpha=0.3)
    plt.fill(exp[:, 0], exp[:, 1], color='salmon', alpha=0.3)
    plt.legend()
    plt.title('识别边界与遍历扩展')
    plt.axis('equal')
    plt.grid(True)
    plt.show()



def main(image_path=None):

    if image_path is None:
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(
            title='选择灰度图像',
            filetypes=[('图像文件', '*.jpg *.png *.bmp *.tif')])
        root.destroy()
        if not image_path:
            raise ValueError('未选择图像文件')

    print(f'读取图像: {image_path}')
    original_image = imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise ValueError(f'无法读取图像: {image_path}')

    if original_image.dtype != np.uint8:
        original_image = cv2.normalize(original_image, None, 0, 255,
                                       cv2.NORM_MINMAX).astype(np.uint8)

    print(f'图像尺寸: {original_image.shape}')
    print('\n执行Sobel边界检测...')
    boundary_image = sobel_boundary_detection(original_image)

    print('\n获取调制区域...')
    region_map, layer_info = get_modulation_region(original_image, boundary_image)

    boundary_poly = extract_boundary_poly(boundary_image)
    if boundary_poly is None:
        raise ValueError('未能从边界图像中提取有效边界多边形')
    print(f'提取到边界多边形顶点数: {len(boundary_poly)}')

    print('\n开始遍历边界...')
    expanded_poly = expand_boundary(boundary_poly, radius=6.0, step=1.0, arc_segments=12, overlap_spacing=0.8)
    print(f'遍历后轮廓点数: {len(expanded_poly)}')

    show_expansion(original_image, boundary_image, region_map, layer_info, boundary_poly, expanded_poly)
    print('\n已完成边界识别、调制区域分层与遍历。')


if __name__ == '__main__':
    main()
