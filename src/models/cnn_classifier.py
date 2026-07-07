import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import pandas as pd
from tqdm import tqdm

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FineTunedResNet50(nn.Module):
    def __init__(self, num_classes=4):
        super(FineTunedResNet50, self).__init__()
        # Load pre-trained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # We need the backbone to be named 'base' so that 'base.7' refers to layer4 (last conv layer)
        # Children 0 to 7 of ResNet50 represent conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        self.base = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Custom classification head
        in_features = 2048
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = torch.cat([x, x, x], dim=1)
        x = self.base(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train_cnn(train_loader, val_loader, epochs=5, lr=0.0001, save_dir="models"):
    """
    Instantiates and trains the FineTunedResNet50 model.
    Matches signature in scripts/train.py.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate model
    model = FineTunedResNet50(num_classes=4)
    
    # Freeze earlier layers (e.g., layers 0 to 5 in self.base)
    # self.base contains [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4]
    # We freeze conv1 up to layer3 (index 0 to 6), only training layer4 (index 7) and fc
    for param in model.base[:7].parameters():
        param.requires_grad = False
        
    model = model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss(gamma=2.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_weights_path = os.path.join(save_dir, "resnet50_best.pth")
    
    print(f"Training CNN on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, targets in train_loader:
            # Duplicate to 3 channels if grayscale
            if images.shape[1] == 1:
                images = torch.cat([images, images, images], dim=1)
                
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                if images.shape[1] == 1:
                    images = torch.cat([images, images, images], dim=1)
                    
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_weights_path)
            print(f"Saved best model checkpoint to {best_weights_path}")
            
    # Load best weights back
    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        
    return model


def evaluate_cnn_and_save(model, loader, save_dir="models"):
    """
    Evaluates the trained CNN on the test loader and saves predictions.
    Matches signature in scripts/train.py.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_targets = []
    all_preds = []
    all_probas = []
    
    print("Evaluating CNN and saving predictions...")
    with torch.no_grad():
        for images, targets in loader:
            if images.shape[1] == 1:
                images = torch.cat([images, images, images], dim=1)
                
            images = images.to(device)
            outputs = model(images)
            
            # Apply softmax to get probabilities
            probas = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_targets.extend(targets.numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probas.append(probas.cpu().numpy())
            
    all_probas = np.concatenate(all_probas, axis=0)
    
    # Prepare predictions DataFrame
    df_preds = pd.DataFrame({
        'true_label': all_targets,
        'predicted_label': all_preds
    })
    
    # Add probability columns
    num_classes = all_probas.shape[1]
    for c in range(num_classes):
        df_preds[f'probability_class_{c}'] = all_probas[:, c]
        
    preds_path = os.path.join(save_dir, "resnet50_predictions.csv")
    df_preds.to_csv(preds_path, index=False)
    print(f"Saved CNN predictions to {preds_path}")
