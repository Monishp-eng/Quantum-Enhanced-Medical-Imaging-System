import os
import sys
sys.path.insert(0, '.')
import argparse
import pickle
import json
import random
import glob
import numpy as np
import torch

from src.preprocessing.image_preprocessor import preprocess_image
from src.features.cnn_features import CNNFeatureExtractor
from src.features.handcrafted import extract_all_handcrafted

# Class index mapping from the model training process
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def parse_args():
    parser = argparse.ArgumentParser(description="Predict brain tumor type from a raw MRI scan image.")
    parser.add_argument(
        "--image-path", 
        type=str, 
        default=None, 
        help="Path to the MRI scan image. If omitted, a random test image from the dataset will be chosen."
    )
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="quantum_a_svm", 
        choices=["baseline_a_svm", "quantum_a_svm", "baseline_b_xgb", "quantum_b_xgb"],
        help="Name of the model classifier to use for inference."
    )
    parser.add_argument(
        "--models-dir", 
        type=str, 
        default="models", 
        help="Directory where model checkpoints are saved."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Select image path if not provided
    if args.image_path is None:
        raw_images = glob.glob(os.path.join("data", "raw", "*", "*", "*.*"))
        if not raw_images:
            print("Error: No images found in data/raw/ directory.")
            print("Please provide a valid image path using --image-path.")
            return
        args.image_path = random.choice(raw_images)
        actual_label = os.path.basename(os.path.dirname(args.image_path))
        print(f"No image path specified. Randomly selected: {args.image_path}")
        print(f"Ground Truth Label: {actual_label.upper()}")
    else:
        print(f"Analyzing specified scan: {args.image_path}")
        actual_label = None

    # Verify model files exist
    model_path = os.path.join(args.models_dir, f"{args.model_name}.pkl")
    scaler_path = os.path.join(args.models_dir, "scaler.pkl")
    feat_sel_path = os.path.join(args.models_dir, "selected_features.json")
    
    for path, desc in [(model_path, "Model classifier"), (scaler_path, "StandardScaler"), (feat_sel_path, "Feature selection mask")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: {desc} not found at {path}. Run train.py first.")

    # 2. Load model, scaler, and features mask
    print(f"Loading {args.model_name} classifier...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    print("Loading StandardScaler...")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    print("Loading optimized feature selection indices...")
    with open(feat_sel_path, "r") as f:
        feat_indices = json.load(f)

    # 3. Preprocess image
    print("Executing preprocessing pipeline (Skull stripping, CLAHE, resizing, normalizing)...")
    preprocessed_img = preprocess_image(args.image_path, size=(224, 224), grayscale=True)
    
    # 4. Extract deep CNN features
    print("Extracting deep CNN representations (ResNet50)...")
    cnn_extractor = CNNFeatureExtractor(use_cuda=torch.cuda.is_available())
    # Expand dims to batch format (1, 1, 224, 224)
    img_tensor = torch.tensor(preprocessed_img).unsqueeze(0).unsqueeze(0)
    cnn_feats = cnn_extractor.extract(img_tensor)

    # 5. Extract handcrafted texture/shape features
    print("Extracting handcrafted texture descriptors (GLCM, HOG, DWT, Intensity)...")
    handcrafted_feats = extract_all_handcrafted(preprocessed_img).reshape(1, -1)

    # 6. Fuse and scale features
    fused_feats = np.concatenate([cnn_feats, handcrafted_feats], axis=1)
    scaled_feats = scaler.transform(fused_feats)

    # 7. Apply feature selection mask if quantum model is chosen
    if "quantum" in args.model_name:
        print(f"Applying QIEO feature mask: reducing from {scaled_feats.shape[1]} to {len(feat_indices)} features...")
        scaled_feats = scaled_feats[:, feat_indices]
    elif "baseline" in args.model_name:
        # Baseline models used top 100 features from Mutual Info / SelectKBest
        # We can extract the selection indices if stored, or default to the baseline subset.
        # Note: baseline features are handled inside their wrappers or training pipelines.
        # For simplicity, if using baseline_a_svm, select first 100 features as trained:
        scaled_feats = scaled_feats[:, :100]

    # 8. Run classification inference
    print("Running classifier inference...")
    pred_idx = model.predict(scaled_feats)[0]
    
    # If classifier wrapper has a custom class mapping, check it
    if hasattr(model, 'classes_'):
        pred_label = CLASS_NAMES[pred_idx]
    else:
        pred_label = CLASS_NAMES[pred_idx]

    # Calculate probabilities if supported
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(scaled_feats)[0]
    elif hasattr(model, "decision_function"):
        # For SVM, decision function outputs scores, we can apply softmin/softmax
        scores = model.decision_function(scaled_feats)[0]
        # Softmax conversion
        probs = np.exp(scores) / np.sum(np.exp(scores))

    print("\n" + "="*40)
    print("           DIAGNOSTIC RESULTS           ")
    print("="*40)
    print(f"Predicted Diagnosis : {pred_label.upper()}")
    if actual_label:
        match_status = "CORRECT" if pred_label.lower() == actual_label.lower() else "INCORRECT"
        print(f"Ground Truth        : {actual_label.upper()} ({match_status})")
    
    if probs is not None:
        print("\nConfidence Scores:")
        for name, prob in zip(CLASS_NAMES, probs):
            indicator = "* (PREDICTED)" if name == pred_label else ""
            print(f"  - {name.capitalize():<12}: {prob*100:6.2f}% {indicator}")
    print("="*40)

if __name__ == "__main__":
    main()
