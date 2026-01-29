#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
3D Visualization of tensors
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib import cm
import argparse
from pathlib import Path

def load_tensor(filepath):
    """Load tensor from .npy or .pt file"""
    path = Path(filepath)
    if path.suffix == '.npy':
        return np.load(path)
    elif path.suffix == '.pt':
        data = torch.load(path, weights_only=False)
        # Handle both raw tensors and dicts with 'tensor_3d' key
        if isinstance(data, dict) and 'tensor_3d' in data:
            return data['tensor_3d'].numpy()
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        else:
            return data
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

def main():
    parser = argparse.ArgumentParser(description='Visualize 3D tensor')
    parser.add_argument('tensor_file', help='Path to tensor file (.npy or .pt)')
    parser.add_argument('--output-dir', default='.', help='Output directory for plots')
    parser.add_argument('--prefix', default='tensor', help='Prefix for output filenames')
    args = parser.parse_args()
    
    print(f"Loading 3D tensor from: {args.tensor_file}")
    tensor_3d = load_tensor(args.tensor_file)
    
    if tensor_3d.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {tensor_3d.shape}")
    
    H, W, D = tensor_3d.shape
    print(f"Tensor shape: {tensor_3d.shape}")
    print(f"  Height: {H}")
    print(f"  Width:  {W}")
    print(f"  Depth:  {D}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    print(f"\nStatistics:")
    print(f"  Min:  {tensor_3d.min():.4f}")
    print(f"  Max:  {tensor_3d.max():.4f}")
    print(f"  Mean: {tensor_3d.mean():.4f}")
    print(f"  Std:  {tensor_3d.std():.4f}")

    # ===== VISUALIZATION 1: Slice Views =====
    print("\n1. Creating slice views...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"3D Tensor Slices ({H}×{W}×{D})", fontsize=16)

# Height slices
axes[0, 0].imshow(tensor_3d[H//4, :, :], cmap='viridis', aspect='auto')
axes[0, 0].set_title(f'Height Slice @ {H//4}')
axes[0, 0].set_xlabel('Depth')
axes[0, 0].set_ylabel('Width')

axes[0, 1].imshow(tensor_3d[H//2, :, :], cmap='viridis', aspect='auto')
axes[0, 1].set_title(f'Height Slice @ {H//2} (Middle)')
axes[0, 1].set_xlabel('Depth')
axes[0, 1].set_ylabel('Width')

axes[0, 2].imshow(tensor_3d[3*H//4, :, :], cmap='viridis', aspect='auto')
axes[0, 2].set_title(f'Height Slice @ {3*H//4}')
axes[0, 2].set_xlabel('Depth')
axes[0, 2].set_ylabel('Width')

# Depth slices
axes[1, 0].imshow(tensor_3d[:, :, D//4], cmap='plasma', aspect='auto')
axes[1, 0].set_title(f'Depth Slice @ {D//4}')
axes[1, 0].set_xlabel('Width')
axes[1, 0].set_ylabel('Height')

axes[1, 1].imshow(tensor_3d[:, :, D//2], cmap='plasma', aspect='auto')
axes[1, 1].set_title(f'Depth Slice @ {D//2} (Middle)')
axes[1, 1].set_xlabel('Width')
axes[1, 1].set_ylabel('Height')

axes[1, 2].imshow(tensor_3d[:, :, 3*D//4], cmap='plasma', aspect='auto')
axes[1, 2].set_title(f'Depth Slice @ {3*D//4}')
axes[1, 2].set_xlabel('Width')
axes[1, 2].set_ylabel('Height')

plt.tight_layout()
plt.savefig(output_dir / f'{args.prefix}_slices.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir}/{args.prefix}_slices.png")
plt.close()

# ===== VISUALIZATION 2: 3D Scatter (Downsampled) =====
print("\n2. Creating 3D scatter plot...")

# Downsample for visualization (every Nth element)
stride_h, stride_w, stride_d = 16, 8, 4
coords = np.mgrid[0:H:stride_h, 0:W:stride_w, 0:D:stride_d]
h_coords = coords[0].flatten()
w_coords = coords[1].flatten()
d_coords = coords[2].flatten()
values = tensor_3d[::stride_h, ::stride_w, ::stride_d].flatten()

# Color by value
norm = Normalize(vmin=values.min(), vmax=values.max())
colors = cm.viridis(norm(values))

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(d_coords, w_coords, h_coords, 
                     c=values, cmap='viridis', 
                     s=10, alpha=0.6, edgecolors='none')

ax.set_xlabel('Depth (128)')
ax.set_ylabel('Width (256)')
ax.set_zlabel('Height (512)')
ax.set_title('3D Tensor Voxel Cloud (Downsampled)\n512×256×128')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('Weight Value')

plt.savefig(output_dir / f'{args.prefix}_3d_scatter.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir}/{args.prefix}_3d_scatter.png")
plt.close()

# ===== VISUALIZATION 3: Isosurface Outline =====
print("\n3. Creating isosurface visualization...")

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create isosurface at different threshold levels
thresholds = [
    tensor_3d.mean() - 2*tensor_3d.std(),
    tensor_3d.mean(),
    tensor_3d.mean() + 2*tensor_3d.std()
]

colors_iso = ['blue', 'green', 'red']
alphas = [0.1, 0.2, 0.3]

for i, (thresh, color, alpha) in enumerate(zip(thresholds, colors_iso, alphas)):
    # Find voxels above threshold
    mask = tensor_3d > thresh
    
    # Downsample
    mask_down = mask[::8, ::4, ::2]
    coords = np.where(mask_down)
    
    if len(coords[0]) > 0:
        ax.scatter(coords[2]*2, coords[1]*4, coords[0]*8,
                  c=color, s=1, alpha=alpha, 
                  label=f'Threshold {i+1} ({thresh:.4f})')

ax.set_xlabel('Depth (128)')
ax.set_ylabel('Width (256)')
ax.set_zlabel('Height (512)')
ax.set_title('3D Tensor Isosurface Levels\n512×256×128')
ax.legend()

plt.savefig(output_dir / f'{args.prefix}_isosurface.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir}/{args.prefix}_isosurface.png")
plt.close()

# ===== VISUALIZATION 4: Depth Stack Animation Frames =====
print("\n4. Creating depth stack animation frames...")

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
fig.suptitle('Depth Layers Stack (512×256 per layer)', fontsize=16)

depths_to_show = np.linspace(0, D-1, 16, dtype=int)

for idx, depth in enumerate(depths_to_show):
    ax = axes[idx // 4, idx % 4]
    im = ax.imshow(tensor_3d[:, :, depth], cmap='twilight', aspect='auto')
    ax.set_title(f'Depth={depth}/{D}')
    ax.axis('off')
    
plt.tight_layout()
plt.savefig(output_dir / f'{args.prefix}_depth_stack.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir}/{args.prefix}_depth_stack.png")
plt.close()

# ===== VISUALIZATION 5: Statistical Analysis per Dimension =====
print("\n5. Creating statistical analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Mean along each axis
mean_along_height = tensor_3d.mean(axis=0)  # [256, 128]
mean_along_width = tensor_3d.mean(axis=1)   # [512, 128]
mean_along_depth = tensor_3d.mean(axis=2)   # [512, 256]

axes[0, 0].imshow(mean_along_height, cmap='coolwarm', aspect='auto')
axes[0, 0].set_title('Mean along Height → [256, 128]')
axes[0, 0].set_xlabel('Depth')
axes[0, 0].set_ylabel('Width')

axes[0, 1].imshow(mean_along_width, cmap='coolwarm', aspect='auto')
axes[0, 1].set_title('Mean along Width → [512, 128]')
axes[0, 1].set_xlabel('Depth')
axes[0, 1].set_ylabel('Height')

axes[1, 0].imshow(mean_along_depth, cmap='coolwarm', aspect='auto')
axes[1, 0].set_title('Mean along Depth → [512, 256]')
axes[1, 0].set_xlabel('Width')
axes[1, 0].set_ylabel('Height')

# Distribution
axes[1, 1].hist(tensor_3d.flatten(), bins=100, log=True, color='purple', alpha=0.7)
axes[1, 1].set_title('Value Distribution (Log Scale)')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Count')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / f'{args.prefix}_statistics.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir}/{args.prefix}_statistics.png")
plt.close()

print("\n" + "=" * 60)
print("Visualization Complete!")
print("=" * 60)
print("\nGenerated files:")
print("  1. tensor_slices.png - Cross-sectional slices")
print("  2. tensor_3d_scatter.png - 3D voxel cloud")
print("  3. tensor_isosurface.png - Isosurface levels")
print("  4. tensor_depth_stack.png - Depth layers animation frames")
print("  5. tensor_statistics.png - Statistical analysis")
print("\nAll visualizations show the 512×256×128 tensor structure!")
if __name__ == "__main__": main()
