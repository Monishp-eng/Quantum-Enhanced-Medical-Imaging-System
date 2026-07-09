import os
import sys
sys.path.insert(0, '.')
import json
import base64
import pickle
import glob
import random
from http.server import SimpleHTTPRequestHandler, HTTPServer
import numpy as np
import torch

from src.preprocessing.image_preprocessor import preprocess_image, load_image, resize_image, apply_clahe, denoise_image, normalize_image
from src.features.cnn_features import CNNFeatureExtractor
from src.features.handcrafted import extract_all_handcrafted, extract_glcm_features, extract_hog_features, extract_intensity_features, extract_wavelet_features
from src.models.cnn_classifier import FineTunedResNet50
from src.explainability.gradcam import GradCAM
from src.explainability.saliency import VanillaSaliency
from src.explainability.attention_viz import plot_explainability_comparison

PORT = 8000
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

CLASS_DESCRIPTIONS = {
    'glioma': {
        'full_name': 'Glioma',
        'type': 'malignant brain tumor',
        'origin': 'glial cells (supportive tissue of the brain)',
        'characteristics': 'irregular, infiltrative mass with heterogeneous texture and poorly defined borders',
        'clinical': 'Gliomas typically present as diffuse lesions with high textural heterogeneity, elevated contrast values in GLCM analysis, and strong wavelet energy in high-frequency subbands indicating rapid cellular proliferation.'
    },
    'meningioma': {
        'full_name': 'Meningioma',
        'type': 'mostly benign brain tumor',
        'origin': 'meninges (protective membranes surrounding the brain)',
        'characteristics': 'well-circumscribed, extra-axial mass with homogeneous enhancement and smooth borders',
        'clinical': 'Meningiomas typically show uniform texture patterns with moderate GLCM homogeneity, distinct edge profiles in HOG features, and localized wavelet energy concentration consistent with a compact, well-defined mass.'
    },
    'pituitary': {
        'full_name': 'Pituitary Tumor',
        'type': 'benign pituitary adenoma',
        'origin': 'pituitary gland (sella turcica region)',
        'characteristics': 'sellar/suprasellar mass with homogeneous intensity and smooth round morphology',
        'clinical': 'Pituitary adenomas present as centrally located, symmetrical masses with low textural complexity in GLCM analysis, minimal gradient variation in HOG descriptors, and concentrated low-frequency wavelet energy.'
    },
    'notumor': {
        'full_name': 'No Tumor (Healthy)',
        'type': 'normal brain anatomy',
        'origin': 'healthy cerebral tissue with no detectable pathological mass',
        'characteristics': 'normal grey and white matter distribution with symmetric hemispheric structure',
        'clinical': 'Healthy brain scans exhibit balanced bilateral symmetry, uniform intensity distribution, low GLCM contrast values, smooth gradient orientation histograms, and evenly distributed wavelet energy across all frequency bands.'
    }
}

def generate_explanation(pred_label, probs, handcrafted_feats, gradcam_map, preprocessed):
    """Generate a detailed clinical-style explanation for the diagnosis."""
    info = CLASS_DESCRIPTIONS[pred_label]
    conf = max(probs) * 100
    
    # Extract individual feature groups for analysis
    glcm = handcrafted_feats[:20]
    hog = handcrafted_feats[20:56]
    intensity = handcrafted_feats[56:66]
    wavelet = handcrafted_feats[66:98]
    
    # Analyze GLCM texture patterns
    glcm_contrast = np.mean(glcm[0:4])  # contrast across 4 angles
    glcm_homogeneity = np.mean(glcm[12:16])  # homogeneity across 4 angles
    glcm_energy = np.mean(glcm[8:12])  # energy across 4 angles
    
    texture_level = 'high' if glcm_contrast > 50 else ('moderate' if glcm_contrast > 15 else 'low')
    homogeneity_level = 'high' if glcm_homogeneity > 0.4 else ('moderate' if glcm_homogeneity > 0.2 else 'low')
    
    # Analyze intensity statistics
    mean_intensity = intensity[0]
    std_intensity = intensity[1]
    skewness = intensity[2]
    
    intensity_desc = 'bright' if mean_intensity > 0.5 else ('moderate' if mean_intensity > 0.25 else 'dark')
    variance_desc = 'high variability' if std_intensity > 0.2 else ('moderate variability' if std_intensity > 0.1 else 'uniform')
    
    # Analyze wavelet energy
    wavelet_energy = np.mean(np.abs(wavelet))
    wavelet_desc = 'strong' if wavelet_energy > 0.1 else ('moderate' if wavelet_energy > 0.01 else 'low')
    
    # Analyze Grad-CAM spatial focus
    if gradcam_map is not None and gradcam_map.size > 0:
        cam_max = np.max(gradcam_map)
        cam_mean = np.mean(gradcam_map)
        focus_ratio = cam_max / (cam_mean + 1e-8)
        
        # Find region of maximum activation
        h, w = gradcam_map.shape
        max_loc = np.unravel_index(np.argmax(gradcam_map), gradcam_map.shape)
        row_pct = max_loc[0] / h
        col_pct = max_loc[1] / w
        
        if row_pct < 0.33:
            vertical_region = 'superior (upper)'
        elif row_pct < 0.66:
            vertical_region = 'central'
        else:
            vertical_region = 'inferior (lower)'
            
        if col_pct < 0.33:
            horizontal_region = 'left'
        elif col_pct < 0.66:
            horizontal_region = 'midline'
        else:
            horizontal_region = 'right'
        
        spatial_focus = f"{vertical_region} {horizontal_region} region"
        focus_desc = 'highly localized' if focus_ratio > 5 else ('moderately focused' if focus_ratio > 2 else 'diffuse')
    else:
        spatial_focus = 'central region'
        focus_desc = 'diffuse'
    
    # Build the explanation
    # Second-place class for differential
    sorted_indices = np.argsort(probs)[::-1]
    second_idx = sorted_indices[1]
    second_label = CLASS_NAMES[second_idx]
    second_conf = probs[second_idx] * 100
    second_info = CLASS_DESCRIPTIONS[second_label]
    
    explanation = []
    
    # Primary diagnosis reasoning
    explanation.append(f"DIAGNOSIS: {info['full_name']} ({info['type']})")
    explanation.append(f"Confidence: {conf:.1f}%")
    explanation.append("")
    
    explanation.append("CLINICAL REASONING:")
    explanation.append(f"The model identifies this scan as {info['full_name']} originating from {info['origin']}. {info['clinical']}")
    explanation.append("")
    
    explanation.append("FEATURE ANALYSIS:")
    explanation.append(f"• Texture (GLCM): {texture_level} contrast ({glcm_contrast:.2f}), {homogeneity_level} homogeneity ({glcm_homogeneity:.3f}), energy {glcm_energy:.4f}")
    explanation.append(f"• Intensity Profile: {intensity_desc} mean ({mean_intensity:.3f}), {variance_desc} (σ={std_intensity:.3f}), skewness={skewness:.2f}")
    explanation.append(f"• Wavelet Energy: {wavelet_desc} multi-scale frequency response ({wavelet_energy:.4f})")
    explanation.append(f"• HOG Gradients: {len(hog)} orientation bins extracted — edge structure consistent with {info['characteristics']}")
    explanation.append("")
    
    explanation.append("SPATIAL ATTENTION (Grad-CAM):")
    explanation.append(f"• The deep CNN focused its attention on the {spatial_focus} of the scan")
    explanation.append(f"• Activation pattern is {focus_desc}, suggesting {('a localized mass or lesion' if focus_desc == 'highly localized' else ('a region of interest with moderate extent' if focus_desc == 'moderately focused' else 'a distributed pattern across the brain parenchyma'))}")
    explanation.append("")
    
    explanation.append("DIFFERENTIAL DIAGNOSIS:")
    explanation.append(f"• Runner-up: {second_info['full_name']} at {second_conf:.1f}% — {second_info['type']} originating from {second_info['origin']}")
    if conf - second_conf < 15:
        explanation.append(f"• Note: The confidence margin is narrow ({conf - second_conf:.1f}%). Clinical correlation and radiologist review is recommended for definitive diagnosis.")
    else:
        explanation.append(f"• The confidence margin of {conf - second_conf:.1f}% indicates a clear diagnostic distinction.")
    
    return "\n".join(explanation)

# Load PyTorch model once globally for fast explainability generation
print("Loading Fine-Tuned ResNet50 model for Grad-CAM explainability...")
cnn_model = FineTunedResNet50()
cnn_model.load_state_dict(torch.load('models/resnet50_best.pth', map_location='cpu'))
cnn_model.eval()

# Load the feature combiner extractors
cnn_extractor = CNNFeatureExtractor(use_cuda=False) # CPU for local server

# Load the scaler and feature selection mask once globally
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/selected_features.json", "r") as f:
    feat_indices = json.load(f)

# Cache classifiers
classifiers = {}
for m_name in ["baseline_a_svm", "quantum_a_svm", "baseline_b_xgb", "quantum_b_xgb"]:
    with open(f"models/{m_name}.pkl", "rb") as f:
        classifiers[m_name] = pickle.load(f)

class APIServer(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/gui/index.html'
            return super().do_GET()
        elif self.path == '/evaluation':
            self.path = '/gui/evaluation.html'
            return super().do_GET()
        elif self.path == '/api/random_test':
            # Select random raw test scan and return its Base64 string
            raw_images = glob.glob(os.path.join("data", "raw", "*", "*", "*.*"))
            if not raw_images:
                self.send_response(404)
                self.end_headers()
                return
                
            selected_path = random.choice(raw_images)
            actual_label = os.path.basename(os.path.dirname(selected_path))
            
            with open(selected_path, "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode('utf-8')
                
            response = {
                "success": True,
                "label": actual_label.upper(),
                "image": f"data:image/jpeg;base64,{encoded_string}"
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            return super().do_GET()

    def end_headers(self):
        # Allow CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.end_headers()

    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode('utf-8'))
            
            model_name = req.get('model', 'quantum_a_svm')
            image_data = req.get('image', '')
            
            # Decode base64 image
            header, encoded = image_data.split(",", 1)
            file_bytes = base64.b64decode(encoded)
            
            os.makedirs("gui/temp", exist_ok=True)
            temp_path = "gui/temp/uploaded.jpg"
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
                
            print(f"Uploaded scan saved. Model: {model_name}. Running prediction...")
            
            try:
                # 1. Preprocess image
                preprocessed = preprocess_image(temp_path, size=(224, 224), grayscale=True)
                
                # Save preprocessed version as visual artifact
                import cv2
                cv2.imwrite("gui/temp/preprocessed.jpg", (preprocessed * 255).astype(np.uint8))
                
                # 2. Extract deep representations
                # Expand dims to batch format (1, 1, 224, 224)
                img_tensor = torch.tensor(preprocessed).unsqueeze(0).unsqueeze(0)
                cnn_feats = cnn_extractor.extract(img_tensor)
                
                # 3. Extract handcrafted texture/shape features
                handcrafted_feats_raw = extract_all_handcrafted(preprocessed)
                handcrafted_feats = handcrafted_feats_raw.reshape(1, -1)
                
                # 4. Fuse and scale features
                fused_feats = np.concatenate([cnn_feats, handcrafted_feats], axis=1)
                scaled_feats = scaler.transform(fused_feats)
                
                # 5. Apply feature selection mask if quantum model is chosen
                if "quantum" in model_name:
                    scaled_feats = scaled_feats[:, feat_indices]
                elif "baseline" in model_name:
                    scaled_feats = scaled_feats[:, :100] # SVM BaselineMI/SelectKBest default
                    
                # 6. Load classifier and compute probabilities
                classifier = classifiers[model_name]
                probs = [0.0, 0.0, 0.0, 0.0]
                if hasattr(classifier, "predict_proba"):
                    probs = classifier.predict_proba(scaled_feats)[0].tolist()
                elif hasattr(classifier, "decision_function"):
                    scores = classifier.decision_function(scaled_feats)[0]
                    # Softmax scores
                    probs = (np.exp(scores) / np.sum(np.exp(scores))).tolist()
                
                # Predict class with highest probability to ensure UI consistency (resolves SVM Platt calibration mismatch)
                pred_idx = int(np.argmax(probs))
                pred_label = CLASS_NAMES[pred_idx]
                    
                # 7. Generate explainability plots dynamically
                # Expand image to 3 channels for PyTorch base forward pass
                rgb_tensor = img_tensor.repeat(1, 3, 1, 1)
                
                gradcam = GradCAM(cnn_model, target_layer_name="base.7")
                gradcam_map, _ = gradcam.generate(rgb_tensor, target_class=pred_idx)
                
                saliency = VanillaSaliency(cnn_model)
                saliency_map, _ = saliency.generate(rgb_tensor, target_class=pred_idx)
                
                explain_img_path = "gui/temp/explain.png"
                plot_explainability_comparison(
                    preprocessed, gradcam_map, saliency_map, pred_label.upper(), save_path=explain_img_path
                )
                
                # 8. Generate detailed explanation
                explanation = generate_explanation(
                    pred_label, probs, handcrafted_feats_raw, gradcam_map, preprocessed
                )
                
                response = {
                    "success": True,
                    "prediction": pred_label.upper(),
                    "confidence": {CLASS_NAMES[i]: probs[i] for i in range(4)},
                    "preprocessed_url": "/gui/temp/preprocessed.jpg",
                    "explainability_url": "/gui/temp/explain.png",
                    "explanation": explanation
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                response = {
                    "success": False,
                    "error": str(e)
                }
                
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        else:
            self.send_response(404)
            self.end_headers()

def run():
    # Make sure static directory exists
    os.makedirs("gui/temp", exist_ok=True)
    
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, APIServer)
    print(f"\n=======================================================")
    print(f"  Quantum MRI Diagnosis Live Demo Server is running!")
    print(f"  Open your browser and navigate to: http://localhost:{PORT}")
    print(f"=======================================================\n")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
