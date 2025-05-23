import numpy as np
import matplotlib.pyplot as plt
import imageio
from typing import Dict, Optional
import os
import tifffile

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
        plt.tight_layout()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        
        # Close figure to free memory
        plt.close(fig)
    
    # Save frames as GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"GIF saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    tensor1 = np.asarray(tifffile.imread('raw_data/c12_SR_w1L-561_t1.tif'))
    maximum = np.percentile(tensor1[tensor1 > 0], 99.999)
    tensor1 = tensor1.clip(min=0.0, max=maximum) / maximum

    tensor2 = np.asarray(tifffile.imread('raw_data/c12_SR_w1L-561_t1.tif'))
    maximum = np.percentile(tensor2[tensor2 > 0], 99.999)
    minimum = np.percentile(tensor2[tensor2 > 0], 30)
    tensor2 = tensor2.clip(min=minimum, max=maximum) / maximum
    combine_tensors_to_gif(
        tensors={
            "Tensor 1": tensor1,
            "Tensor 2": tensor2
        },
        output_path="combined_tensors.gif",
        fps=2,
        cmap="grey"
    )    