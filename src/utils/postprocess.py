import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def denormalize_image(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    if len(tensor.shape) == 3 and tensor.shape[0] <= 4:  
        tensor = np.transpose(tensor, (1, 2, 0))
    
    mean = np.array(mean)
    std = np.array(std)
    
    denorm_tensor = tensor * std + mean
    
    denorm_tensor = np.clip(denorm_tensor, 0, 1)
    
    return denorm_tensor

def postprocess_output(output, threshold=0.5, output_type='mask'):
   
    if len(output.shape) == 3 and output.shape[0] == 1:
        output = output.squeeze(0)  # (1, H, W) -> (H, W)
    elif len(output.shape) == 3 and output.shape[0] <= 4:
        output = np.transpose(output, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        if output.shape[-1] == 1:
            output = output.squeeze(-1)  # (H, W, 1) -> (H, W)
    
    if output_type == 'mask':
        binary_mask = (output > threshold).astype(np.uint8) * 255
        
        if len(binary_mask.shape) == 2:
            rgb_image = np.stack([binary_mask] * 3, axis=-1)
        else:
            rgb_image = binary_mask
            
    elif output_type == 'heatmap':
        if len(output.shape) == 2:
            heatmap = output
        else:
            heatmap = output
            
        import matplotlib.cm as cm
        colormap = cm.get_cmap('jet')
        rgb_image = colormap(heatmap)[:, :, :3] 
        rgb_image = (rgb_image * 255).astype(np.uint8)
        
    elif output_type == 'colored_mask':
        binary_mask = (output > threshold).astype(np.uint8)
        rgb_image = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
        
        rgb_image[binary_mask == 1] = [0, 100, 255]  # Mavi
        
    return rgb_image



def visualize_results(cam, point_cloud, mask, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(cam)
    axes[0, 0].set_title('Kamera Görüntüsü', fontsize=14, )
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(point_cloud)
    axes[0, 1].set_title('Lidar Görüntüsü', fontsize=14)
    axes[0, 1].axis('off')
    
    if len(mask.shape) == 2:
        axes[1, 0].imshow(mask, cmap='gray')
    else:
        axes[1, 0].imshow(mask)
    axes[1, 0].set_title('Gerçek Maske', fontsize=14)
    axes[1, 0].axis('off')
    
    if len(output.shape) == 2:
        axes[1, 1].imshow(output, cmap='gray')
    else:
        axes[1, 1].imshow(output)
    axes[1, 1].set_title('Tahmin', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_overlay(image, mask, color=(255, 0, 0), opacity=0.5, threshold=0.5):
 
    
    if isinstance(image, Image.Image):
        image = np.array(image).copy()
    elif isinstance(image, torch.Tensor):
        image = denormalize_image(image)
        image = (image * 255).astype(np.uint8)
    else:
        image = image.copy()
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    elif len(mask.shape) == 3 and mask.shape[0] <= 4:
        mask = np.transpose(mask, (1, 2, 0))
        if mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
    
    if mask.max() > 1.0:
        mask = mask / mask.max()
    
    binary_mask = (mask > threshold)
    
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    if image.shape[:2] != binary_mask.shape[:2]:
        from scipy.ndimage import zoom
        scale_y = image.shape[0] / binary_mask.shape[0]
        scale_x = image.shape[1] / binary_mask.shape[1]
        binary_mask = zoom(binary_mask.astype(float), (scale_y, scale_x), order=0) > 0.5
    
    overlay_image = image.copy()
    
    if np.any(binary_mask):
        for channel in range(3):
            original_values = image[binary_mask, channel].astype(np.float32)
            color_values = np.full_like(original_values, color[channel], dtype=np.float32)
            
            blended_values = (1 - opacity) * original_values + opacity * color_values
            overlay_image[binary_mask, channel] = np.clip(blended_values, 0, 255).astype(np.uint8)
    
    return overlay_image


def visualize_overlay_comparison(image, mask, color=(255, 0, 0), opacity=0.5, threshold=0.5):
    
    overlay_image = visualize_overlay(image, mask, color, opacity, threshold)
    
    if isinstance(image, torch.Tensor):
        original_image = denormalize_image(image)
        original_image = (original_image * 255).astype(np.uint8)
    else:
        original_image = image
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Orijinal Görüntü', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(overlay_image)
    axes[1].set_title(f'Overlay (Opacity: {opacity}, Color: {color})', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return overlay_image

