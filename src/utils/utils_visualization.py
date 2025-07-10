import re
import os
import imageio
import numpy as np
from typing import List, Dict, Optional
import matplotlib.pyplot as plt


def combine_tensors_to_gif(
    tensors: Dict[str, np.ndarray],
    output_path: str = "combined_tensors.gif",
    fps: int = 2,
    cmap: str = "viridis",
    figsize: tuple = (12, 8),
    dpi: int = 100,
    show_colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> None:

    # Check all tensors have the same depth
    depths = [tensor.shape[0] for tensor in tensors.values()]   
    depth = max(depths)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate frames for GIF
    frames = []
    
    for i in range(depth):
        # Create figure and axes
        fig, axes = plt.subplots(1, len(tensors), figsize=figsize, dpi=dpi, sharey=True)
        if len(tensors) == 1:
            axes = [axes]  # Ensure axes is always a list
        
        # Plot each tensor's current layer
        ims = []
        for j, (name, tensor) in enumerate(tensors.items()):
            im = axes[j].imshow(tensor[i % tensor.shape[0]], cmap=cmap, vmin=vmin, vmax=vmax)
            ims.append(im)
            axes[j].set_title(f"{name} - Layer {i % tensor.shape[0]}")
        
        # Add colorbar if specified
        if show_colorbar:
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(ims[0], cax=cbar_ax)
        
        # Render figure to numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        
        # Close figure to free memory
        plt.close(fig)
    
    # Save frames as GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"GIF saved to: {output_path}")


def parse_log_file(log_file_path: str) -> Dict[str, List[float]]:
    metrics = {'loss': [],
               'lr': [],
               'vali_loss': [],
               'train_loss': []}
    patterns = {'loss': r'\[loss: ([\d.e+-]+)\]', 
                'lr': r'\[lr: ([\d.e+-]+)\]', 
                'vali_loss': r'\[vali_loss: ([\d.e+-]+)\]', 'train_loss': r'\[train_loss: ([\d.e+-]+)\]'}  
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                
                for metric, pattern in patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        value = float(match.group(1))
                        metrics[metric].append(value)
    except Exception as e:
        print(f"error: {e}")
    return metrics


def plot_loss_function(metrics: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # loss curve
    if metrics['loss']:
        plt.plot(metrics['loss'], label='Training Loss', alpha=0.7)
    
    if metrics['vali_loss']:
        vali_x = [i for i in range(len(metrics['loss'])) if i % (len(metrics['loss']) // len(metrics['vali_loss'])) == 0][:len(metrics['vali_loss'])]
        plt.plot(vali_x, metrics['vali_loss'], label='Validation Loss', marker='o', markersize=4)
    
    if metrics['train_loss']:
        train_x = [i for i in range(len(metrics['loss'])) if i % (len(metrics['loss']) // len(metrics['train_loss'])) == 0][:len(metrics['train_loss'])]
        plt.plot(train_x, metrics['train_loss'], label='Epoch Train Loss', marker='s', markersize=4)
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # lr curve
    plt.subplot(1, 2, 2)
    if metrics['lr']:
        plt.plot(metrics['lr'], label='Learning Rate', color='orange')
        plt.xlabel('Iterations')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('loss_plot.png')
    plt.show()


def plot_tensor_histogram(tensor: np.ndarray, 
                          title: str = "Histogram of tensor",
                          output_path: Optional[str] = None,
                          bins: int = 100,
                          range_min: int = 0,
                          range_max: int = 1,
                          color: str = 'blue') -> None:

    plt.figure(figsize=(10, 6))
    flat_tensor = tensor.flatten()
    plt.hist(flat_tensor, bins=bins, range=[range_min, range_max], 
             color=color, alpha=0.7)
    
    plt.title(title, fontsize=14)
    plt.xlabel("Intensity", fontsize=12)
    plt.ylabel("Freqs", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Histograms saved to: {output_path}")
    plt.close()


def plot_multiple_tensors(tensors: List[np.ndarray],
                          titles: List[str] = None,
                          output_path: Optional[str] = None,
                          bins: int = 100,
                          range_min: int = 0,
                          range_max: int = 1,
                          colors: List[str] = None) -> None:
    
    num_tensors = len(tensors)
    if titles is None:
        titles = [f"Tensor {i+1} intensity distribution" for i in range(num_tensors)]
    
    # 设置默认颜色
    if colors is None:
        default_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
        colors = [default_colors[i % len(default_colors)] for i in range(num_tensors)]
    
    if num_tensors <= 3:
        nrows, ncols = 1, num_tensors
        figsize = (5 * num_tensors, 5)
    else:
        nrows = int(np.ceil(num_tensors / 3))
        ncols = 3 if num_tensors >= 3 else num_tensors
        figsize = (15, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, tensor in enumerate(tensors):
        row, col = i // ncols, i % ncols
        ax = axes[row, col]
        flat_tensor = tensor.flatten()
        ax.hist(flat_tensor, bins=bins, range=[range_min, range_max], 
                color=colors[i], alpha=0.7)
        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel("Intensity", fontsize=10)
        ax.set_ylabel("Freqs", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
    for i in range(num_tensors, nrows * ncols):
        row, col = i // ncols, i % ncols
        axes[row, col].axis('off')

    if output_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"histograms saved to: {output_path}")
    plt.close()




if __name__ == "__main__":
    log_file_path = '/home/tyche/paddle_SN2N/data/data_2d/logs/output_2025-07-08 10:03:40.451213.log'
    metrics = parse_log_file(log_file_path)
    for metric, values in metrics.items():
        print(f"{metric}: {values[:5]}... ({len(values)} values)")
    plot_loss_function(metrics)

