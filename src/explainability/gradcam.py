import torch
import torch.nn as nn
import numpy as np

class GradCAM:
    """
    Grad-CAM class to generate attention heatmaps from the last conv layer of ResNet50.
    """
    def __init__(self, model, target_layer_name="base.7"): # Layer 7 is the last bottleneck block (layer4) in Sequential base
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # Find target layer
        self.target_layer = None
        for name, module in self.model.named_modules():
            if name == target_layer_name:
                self.target_layer = module
                break
                
        if self.target_layer is None:
            raise ValueError(f"Could not find module with name {target_layer_name} in model.")
            
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        """Generates Grad-CAM heatmap."""
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            # Default to class with highest probability
            target_class = torch.argmax(output, dim=1).item()
            
        # Backward pass
        score = output[0, target_class]
        score.backward()
        
        # Pull gradients and activations
        gradients = self.gradients.cpu().data.numpy()      # Shape: (1, 2048, H_conv, W_conv)
        activations = self.activations.cpu().data.numpy()  # Shape: (1, 2048, H_conv, W_conv)
        
        # Mean of gradients per channel (weights)
        weights = np.mean(gradients, axis=(2, 3))[0]       # Shape: (2048,)
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[2:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[0, i]
            
        # Apply ReLU to retain only positive contributions
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam, target_class
