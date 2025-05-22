import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

def visualize_large_tensor(tensor, chunk_size=1, title="Tensor Visualization", interval=1000, cmap="grey"):
    layers, rows, cols = tensor.shape
    
    # 创建图形（仅一个子图）
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(title, fontsize=16)
    
    vmin, vmax = 0, 255  # 示例值，根据实际数据调整
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # 初始化图像
    im = ax.imshow(np.zeros((rows, cols)), cmap=cmap, norm=norm)
    ax.set_title("Current Layer")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    fig.colorbar(im, ax=ax)
    
    layer_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                         color='white', fontweight='bold')
    
    # 动态更新函数
    def update(frame):
        # 仅加载当前需要的层（而非整个张量）
        current_layer = tensor[frame].astype(np.float32)
        
        # 更新2D图像
        im.set_data(current_layer)
        layer_text.set_text(f"Layer {frame+1}/{layers}")
        ax.set_title(f"Layer {frame+1}: Value Range [{current_layer.min():.2f}, {current_layer.max():.2f}]")
        
        return im, layer_text
    
    # 创建动画（禁用帧缓存）
    ani = FuncAnimation(fig, update, frames=layers, interval=interval, 
                        blit=True, cache_frame_data=False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return ani

if __name__ == "__main__":
    # 创建一个大型随机张量并保存为内存映射文件
    tensor = np.array(tifffile.imread('c12_SR_w1L-561_t1.tif'))
    
    # 可视化大型张量
    ani = visualize_large_tensor(tensor, chunk_size=1, interval=500)
    plt.show()
    ani.save("result.gif", writer='pillow', fps=24)