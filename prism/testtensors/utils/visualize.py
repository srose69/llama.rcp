#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Shield-1.0.0
# Copyright (c) 2026 Ivan K
# Co-licensed under LLaMa.RCP project terms
# @ai-training prohibited
#
# Source: https://github.com/srose69/llama.rcp

"""
Flexible PRISM visualization tool for pipes, workers, and tiles
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from pathlib import Path


def load_pipe(pipe_path):
    """Load pipe state dict"""
    return torch.load(pipe_path, map_location='cpu', weights_only=False)


def visualize_tile(tile, title="Tile", ax=None, cmap='viridis'):
    """Visualize a single tile (2D heatmap)"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(tile, cmap=cmap, aspect='auto')
    ax.set_title(f'{title}\nShape: {tile.shape}, Range: [{tile.min():.4f}, {tile.max():.4f}]')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')
    plt.colorbar(im, ax=ax, label='Weight Value')
    
    return ax


def visualize_worker(pipe_state, worker_id, title_prefix="Worker"):
    """Visualize a single worker's tile with statistics"""
    worker_key = f'workers.{worker_id}.tile'
    if worker_key not in pipe_state:
        print(f"❌ Worker {worker_id} not found!")
        return
    
    tile = pipe_state[worker_key].numpy()
    
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 3, width_ratios=[2, 1, 1])
    
    # Heatmap
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(tile, cmap='viridis', aspect='auto')
    ax1.set_title(f'{title_prefix} {worker_id}\nShape: {tile.shape}')
    ax1.set_xlabel('Columns')
    ax1.set_ylabel('Rows')
    plt.colorbar(im, ax=ax1, label='Weight Value')
    
    # Distribution
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(tile.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax2.set_title('Value Distribution')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    
    # Statistics
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    stats_text = f"""
    Statistics:
    
    Min:  {tile.min():.6f}
    Max:  {tile.max():.6f}
    Mean: {tile.mean():.6f}
    Std:  {tile.std():.6f}
    
    Shape: {tile.shape}
    Size:  {tile.size:,}
    
    Zeros: {(tile == 0).sum():,}
    ({(tile == 0).sum() / tile.size * 100:.2f}%)
    """
    ax3.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    return fig


def visualize_pipe_overview(pipe_state, pipe_name="Pipe"):
    """Visualize overview of all workers in a pipe"""
    worker_keys = sorted([k for k in pipe_state.keys() if 'workers' in k and 'tile' in k])
    
    if not worker_keys:
        print(f"❌ No workers found in {pipe_name}")
        return
    
    num_workers = len(worker_keys)
    print(f"📊 Visualizing {num_workers} workers from {pipe_name}...")
    
    # Grid layout
    cols = int(np.ceil(np.sqrt(num_workers)))
    rows = int(np.ceil(num_workers / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(f'{pipe_name} - All Workers ({num_workers} total)', fontsize=16)
    
    if num_workers == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, worker_key in enumerate(worker_keys):
        tile = pipe_state[worker_key].numpy()
        ax = axes[idx]
        
        im = ax.imshow(tile, cmap='viridis', aspect='auto')
        worker_id = worker_key.split('.')[1]
        ax.set_title(f'W{worker_id}', fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_workers, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_pipe_statistics(pipe_state, pipe_name="Pipe"):
    """Visualize statistical overview of pipe"""
    worker_keys = sorted([k for k in pipe_state.keys() if 'workers' in k and 'tile' in k])
    router_keys = [k for k in pipe_state.keys() if 'router' in k]
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3)
    
    # Collect all worker statistics
    all_mins = []
    all_maxs = []
    all_means = []
    all_stds = []
    all_values = []
    
    for worker_key in worker_keys:
        tile = pipe_state[worker_key].numpy()
        all_mins.append(tile.min())
        all_maxs.append(tile.max())
        all_means.append(tile.mean())
        all_stds.append(tile.std())
        all_values.extend(tile.flatten().tolist())
    
    # Plot 1: Min/Max ranges per worker
    ax1 = fig.add_subplot(gs[0, :])
    worker_ids = range(len(worker_keys))
    ax1.plot(worker_ids, all_mins, label='Min', alpha=0.7)
    ax1.plot(worker_ids, all_maxs, label='Max', alpha=0.7)
    ax1.fill_between(worker_ids, all_mins, all_maxs, alpha=0.3)
    ax1.set_title(f'{pipe_name} - Weight Ranges per Worker')
    ax1.set_xlabel('Worker ID')
    ax1.set_ylabel('Weight Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean per worker
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(worker_ids, all_means, alpha=0.7, color='green')
    ax2.set_title('Mean per Worker')
    ax2.set_xlabel('Worker ID')
    ax2.set_ylabel('Mean Value')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Std per worker
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(worker_ids, all_stds, alpha=0.7, color='orange')
    ax3.set_title('Std Dev per Worker')
    ax3.set_xlabel('Worker ID')
    ax3.set_ylabel('Std Dev')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Overall distribution
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(all_values, bins=100, color='purple', alpha=0.7, edgecolor='black')
    ax4.set_title('Overall Value Distribution')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Count')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Router weights heatmap (if exists)
    if router_keys:
        router_weight_key = [k for k in router_keys if 'weight' in k][0]
        router_weights = pipe_state[router_weight_key].numpy()
        
        ax5 = fig.add_subplot(gs[2, :])
        im = ax5.imshow(router_weights, cmap='RdBu_r', aspect='auto')
        ax5.set_title(f'Router Weights {router_weights.shape}')
        ax5.set_xlabel('Input Dimension')
        ax5.set_ylabel('Output (Worker Selection)')
        plt.colorbar(im, ax=ax5, label='Weight Value')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize PRISM pipes, workers, and tiles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize entire Pipe 1
  python visualize.py --pipe 1
  
  # Visualize Pipe 2 worker 100
  python visualize.py --pipe 2 --worker 100
  
  # Visualize specific tile file
  python visualize.py --tile path/to/tile.npy
  
  # Show plot instead of saving
  python visualize.py --pipe 1 --show
        """
    )
    
    parser.add_argument('--pipe', type=int, choices=[1, 2, 3],
                        help='Pipe number to visualize (1, 2, or 3)')
    parser.add_argument('--worker', type=int,
                        help='Specific worker ID to visualize')
    parser.add_argument('--tile', type=str,
                        help='Path to specific tile file (.npy or .pt)')
    parser.add_argument('--show', action='store_true',
                        help='Show plot interactively instead of saving')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Output directory for saved plots')
    parser.add_argument('--pipes-dir', type=str, default='builded_pipes',
                        help='Directory containing pipe .pt files')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.pipe, args.tile]):
        parser.error("Must specify --pipe or --tile")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize specific tile
    if args.tile:
        print(f"📊 Loading tile from: {args.tile}")
        tile_path = Path(args.tile)
        if tile_path.suffix == '.npy':
            tile = np.load(tile_path)
        elif tile_path.suffix == '.pt':
            tile = torch.load(tile_path, weights_only=False)
            if isinstance(tile, torch.Tensor):
                tile = tile.numpy()
        else:
            print(f"❌ Unsupported format: {tile_path.suffix}")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        visualize_tile(tile, title=f"Tile: {tile_path.name}", ax=ax)
        
        if args.show:
            plt.show()
        else:
            output_file = output_dir / f"{tile_path.stem}_visualization.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {output_file}")
        
        return
    
    # Visualize pipe
    if args.pipe:
        pipe_path = Path(args.pipes_dir) / f"pipe{args.pipe}.pt"
        if not pipe_path.exists():
            print(f"❌ Pipe file not found: {pipe_path}")
            return
        
        print(f"📊 Loading Pipe {args.pipe} from: {pipe_path}")
        pipe_state = load_pipe(pipe_path)
        
        # Visualize specific worker
        if args.worker is not None:
            fig = visualize_worker(pipe_state, args.worker, 
                                   title_prefix=f"Pipe {args.pipe} Worker")
            
            if args.show:
                plt.show()
            else:
                output_file = output_dir / f"pipe{args.pipe}_worker{args.worker}.png"
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"✅ Saved: {output_file}")
        
        # Visualize entire pipe
        else:
            # Overview
            fig1 = visualize_pipe_overview(pipe_state, pipe_name=f"Pipe {args.pipe}")
            if args.show:
                plt.show()
            else:
                output_file = output_dir / f"pipe{args.pipe}_overview.png"
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"✅ Saved: {output_file}")
                plt.close(fig1)
            
            # Statistics
            fig2 = visualize_pipe_statistics(pipe_state, pipe_name=f"Pipe {args.pipe}")
            if args.show:
                plt.show()
            else:
                output_file = output_dir / f"pipe{args.pipe}_statistics.png"
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"✅ Saved: {output_file}")
                plt.close(fig2)
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
