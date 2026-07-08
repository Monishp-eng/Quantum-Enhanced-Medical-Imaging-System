import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from .cnn_features import CNNFeatureExtractor
from .handcrafted import extract_all_handcrafted

class FeatureCombiner:
    """Fuses CNN and Handcrafted features and applies standardization."""
    def __init__(self, use_cuda=True):
        self.cnn_extractor = CNNFeatureExtractor(use_cuda=use_cuda)
        self.scaler = StandardScaler()

    def combine_features_for_batch(self, images_tensor):
        """
        Extracts and combines features for a batch of image tensors.
        images_tensor shape: (B, 1, 224, 224) or (B, 3, 224, 224)
        """
        batch_size = images_tensor.shape[0]
        
        # 1. Extract CNN features (B, 2048)
        cnn_feats = self.cnn_extractor.extract(images_tensor)
        
        # 2. Extract Handcrafted features for each image in the batch (B, 98) in parallel
        # Convert images back to numpy for handcrafted extraction
        images_np = images_tensor.cpu().numpy()
        from joblib import Parallel, delayed
        handcrafted_feats_list = Parallel(n_jobs=-1, prefer="threads")(
            delayed(extract_all_handcrafted)(images_np[i, 0]) for i in range(batch_size)
        )
        handcrafted_feats = np.stack(handcrafted_feats_list, axis=0)
        
        # 3. Concatenate CNN and Handcrafted features -> (B, 2146)
        combined = np.concatenate([cnn_feats, handcrafted_feats], axis=1)
        return combined

    def process_and_save(self, train_loader, val_loader, test_loader, save_dir):
        """
        Extracts combined features from training, validation, and test sets.
        Saves standardized matrices to save_dir.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        datasets = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        extracted_data = {}
        
        for split, loader in datasets.items():
            print(f"\nProcessing {split} split...")
            features_list = []
            labels_list = []
            
            for images, labels in tqdm(loader):
                feats = self.combine_features_for_batch(images)
                features_list.append(feats)
                labels_list.append(labels.numpy())
                
            X = np.concatenate(features_list, axis=0)
            y = np.concatenate(labels_list, axis=0)
            extracted_data[split] = (X, y)
            
        # Fit scaler on training set, then transform all sets
        X_train, y_train = extracted_data['train']
        X_val, y_val = extracted_data['val']
        X_test, y_test = extracted_data['test']
        
        print("\nStandardizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save matrices
        np.save(os.path.join(save_dir, "X_train.npy"), X_train_scaled)
        np.save(os.path.join(save_dir, "y_train.npy"), y_train)
        np.save(os.path.join(save_dir, "X_val.npy"), X_val_scaled)
        np.save(os.path.join(save_dir, "y_val.npy"), y_val)
        np.save(os.path.join(save_dir, "X_test.npy"), X_test_scaled)
        np.save(os.path.join(save_dir, "y_test.npy"), y_test)
        
        # Save scaler for inference deployment
        import pickle
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        
        print(f"Features and scaler saved successfully to {save_dir}")
        print(f"Feature matrix dimensions: {X_train_scaled.shape}")


def extract_and_combine_features(train_loader, test_loader, save_dir):
    """
    Notebook compatibility wrapper.
    Instantiates FeatureCombiner, processes train and test loaders, and returns the matrices.
    """
    import os
    combiner = FeatureCombiner()
    # We pass test_loader for validation split to satisfy the three-split loader requirements
    combiner.process_and_save(train_loader, test_loader, test_loader, save_dir)
    
    X_train = np.load(os.path.join(save_dir, "X_train.npy"))
    y_train = np.load(os.path.join(save_dir, "y_train.npy"))
    X_test = np.load(os.path.join(save_dir, "X_test.npy"))
    y_test = np.load(os.path.join(save_dir, "y_test.npy"))
    return X_train, y_train, X_test, y_test

