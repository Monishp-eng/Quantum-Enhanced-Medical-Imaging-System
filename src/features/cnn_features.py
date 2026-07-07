import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

class CNNFeatureExtractor:
    """CNN-based feature extractor using a pretrained ResNet50 model."""
    def __init__(self, use_cuda=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        print(f"CNN Feature Extractor using device: {self.device}")
        
        # Load pretrained ResNet50 (with updated weights arg to prevent deprecation warnings)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # We need the 2048-dim vector from the avgpool layer, just before the FC layer.
        self.extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.extractor = self.extractor.to(self.device)
        self.extractor.eval()

    def extract(self, image_tensor):
        """
        Extracts features from a single image or a batch of images.
        Input image_tensor shape: (B, 3, 224, 224) or (B, 1, 224, 224)
        """
        # If input has only 1 channel, duplicate it to 3 channels for ResNet
        if image_tensor.shape[1] == 1:
            image_tensor = torch.cat([image_tensor, image_tensor, image_tensor], dim=1)
            
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            features = self.extractor(image_tensor)
            # Squeeze out spatial dimensions (B, 2048, 1, 1) -> (B, 2048)
            features = torch.squeeze(features)
            # If batch size is 1, squeeze makes it 1D, we want to maintain (B, 2048)
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
                
        return features.cpu().numpy()

    def extract_dataset(self, dataloader):
        """Extracts features for an entire PyTorch DataLoader."""
        all_features = []
        all_labels = []
        
        print("Extracting CNN features...")
        for images, labels in tqdm(dataloader):
            feats = self.extract(images)
            all_features.append(feats)
            all_labels.append(labels.numpy())
            
        import numpy as np
        return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)
