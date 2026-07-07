import torch
import numpy as np

class VanillaSaliency:
    """
    Vanilla Saliency map generation which computes the gradients of the target class score
    with respect to the input image.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate(self, input_tensor, target_class=None):
        """
        Generates Saliency map.
        input_tensor shape: (1, C, H, W)
        """
        # Enable gradients on the input image
        input_tensor.requires_grad_()
        
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        # Backward pass
        score = output[0, target_class]
        score.backward()
        
        # Pull gradients with respect to the input image
        saliency = input_tensor.grad.data.abs() # Take absolute value
        
        # Take the maximum across color channels (if RGB)
        saliency, _ = torch.max(saliency, dim=1)
        saliency = saliency.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        if saliency.max() > 0:
            saliency = saliency / saliency.max()
            
        return saliency, target_class
