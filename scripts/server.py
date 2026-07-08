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
from src.features.handcrafted import extract_all_handcrafted
from src.models.cnn_classifier import FineTunedResNet50
from src.explainability.gradcam import GradCAM
from src.explainability.saliency import VanillaSaliency
from src.explainability.attention_viz import plot_explainability_comparison

PORT = 8000
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

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
                handcrafted_feats = extract_all_handcrafted(preprocessed).reshape(1, -1)
                
                # 4. Fuse and scale features
                fused_feats = np.concatenate([cnn_feats, handcrafted_feats], axis=1)
                scaled_feats = scaler.transform(fused_feats)
                
                # 5. Apply feature selection mask if quantum model is chosen
                if "quantum" in model_name:
                    scaled_feats = scaled_feats[:, feat_indices]
                elif "baseline" in model_name:
                    scaled_feats = scaled_feats[:, :100] # SVM BaselineMI/SelectKBest default
                    
                # 6. Load classifier and predict
                classifier = classifiers[model_name]
                pred_idx = classifier.predict(scaled_feats)[0]
                pred_label = CLASS_NAMES[pred_idx]
                
                # Compute probabilities
                probs = [0.0, 0.0, 0.0, 0.0]
                if hasattr(classifier, "predict_proba"):
                    probs = classifier.predict_proba(scaled_feats)[0].tolist()
                elif hasattr(classifier, "decision_function"):
                    scores = classifier.decision_function(scaled_feats)[0]
                    # Softmax scores
                    probs = (np.exp(scores) / np.sum(np.exp(scores))).tolist()
                    
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
                
                response = {
                    "success": True,
                    "prediction": pred_label.upper(),
                    "confidence": {CLASS_NAMES[i]: probs[i] for i in range(4)},
                    "preprocessed_url": "/temp/preprocessed.jpg",
                    "explainability_url": "/temp/explain.png"
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
