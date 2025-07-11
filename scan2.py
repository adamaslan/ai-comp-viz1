#!/usr/bin/env python3
"""
Minimal, efficient pipeline to detect notebook drawings in images using MiDaS small and Open3D.
Usage:
    python detect_notebook_drawing.py <image_or_folder> [--visualize]
"""
import sys
sys.argv = ["detect_notebook_drawing.py", "path/to/your/image.png", "--visualize"]

import argparse
import argparse
import os
import torch
import cv2
import numpy as np
import open3d as o3d


def load_model(device):
    # Load MiDaS small model and transforms
    model = torch.hub.load('intel-isl/MiDaS', 'DPT_Small')
    model.to(device).eval()
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    transform = transforms.small_transform
    return model, transform


def estimate_depth(model, transform, img, device):
    # Prepare and run depth estimation
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()
    return prediction.cpu().numpy()


def generate_pointcloud(depth, sample_stride=4):
    # Downsample depth map for speed
    h, w = depth.shape
    fx = fy = 500.0  # Approximate focal length
    cx, cy = w / 2, h / 2

    ys, xs = np.mgrid[::sample_stride, ::sample_stride]
    z = depth[ys, xs]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def analyze_pcd(pcd):
    # Compute flatness and size heuristics
    bbox = pcd.get_axis_aligned_bounding_box()
    dims = bbox.extent
    z_vals = np.asarray(pcd.points)[:, 2]
    flatness = float(np.std(z_vals))
    is_flat = flatness < 0.01
    is_small = dims[0] < 0.3 and dims[1] < 0.3
    return is_flat and is_small, flatness, dims


def process_image(path, model, transform, device, visualize=False):
    # Load and preprocess image
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))

    # Depth estimation and point cloud generation
    depth = estimate_depth(model, transform, img, device)
    pcd = generate_pointcloud(depth)

    # Analysis
    result, flatness, dims = analyze_pcd(pcd)

    if visualize:
        o3d.visualization.draw_geometries([pcd])
    return result, flatness, dims


def main():
    parser = argparse.ArgumentParser(description="Detect notebook drawings in images.")
    parser.add_argument('input', help='Path to an image file or folder of images')
    parser.add_argument('--visualize', action='store_true', help='Visualize the point cloud')
    args = parser.parse_args()

    # Select device (MPS for Mac M2 or CPU)
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model, transform = load_model(device)

    # Gather input files
    paths = []
    if os.path.isdir(args.input):
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        for fname in os.listdir(args.input):
            if os.path.splitext(fname)[1].lower() in extensions:
                paths.append(os.path.join(args.input, fname))
    else:
        paths = [args.input]

    # Process each image
    for path in paths:
        try:
            result, flatness, dims = process_image(path, model, transform, device, args.visualize)
            status = 'Likely' if result else 'Unlikely'
            print(f"{path}: {status} a notebook drawing (flatness={flatness:.4f}, size={dims})")
        except Exception as e:
            print(f"{path}: Error - {e}")

if __name__ == '__main__':
    main()
