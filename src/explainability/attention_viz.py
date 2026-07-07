import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlay_heatmap_on_image(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays a normalized heatmap [0, 1] on a normalized image [0, 1].
    """
    # Convert image to uint8 [0, 255]
    img_uint8 = (image * 255).astype(np.uint8)
    if len(img_uint8.shape) == 2:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
    
    # Scale heatmap to [0, 255] and apply colormap
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Overlay
    overlay = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Convert back to float [0, 1] for matplotlib plotting
    return overlay.astype(np.float32) / 255.0

def plot_explainability_comparison(image, gradcam_map, saliency_map, target_class_name, save_path=None):
    """
    Plots a side-by-side comparison: Original / Grad-CAM / Saliency.
    """
    overlay = overlay_heatmap_on_image(image, gradcam_map)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original MRI')
    axes[0].axis('off')
    
    # Grad-CAM
    axes[1].imshow(overlay)
    axes[1].set_title(f'Grad-CAM Overlay ({target_class_name})')
    axes[1].axis('off')
    
    # Saliency
    axes[2].imshow(saliency_map, cmap='hot')
    axes[2].set_title(f'Saliency Map ({target_class_name})')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
