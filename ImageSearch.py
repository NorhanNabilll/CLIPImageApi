from flask import Flask, request, jsonify
import os
import torch
import clip
from PIL import Image
import numpy as np
import base64
import io
import json
import warnings
from flask_cors import CORS

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

class CLIPImageSearchModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.model.eval()
            self.initialized = True
        except Exception as e:
            self.initialized = False
            self.error_message = str(e)

    def extract_image_features(self, image_base64):
        if not self.initialized:
            return None, f"Model not initialized: {self.error_message}"
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten().tolist(), None
        except Exception as e:
            return None, f"Error processing image: {str(e)}"

    def calculate_similarity(self, features1, features2):
        try:
            features1 = np.array(features1)
            features2 = np.array(features2)
            similarity = np.dot(features1, features2)
            return float(similarity), None
        except Exception as e:
            return None, f"Error calculating similarity: {str(e)}"

# Initialize the model globally
search_model = CLIPImageSearchModel()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_initialized": search_model.initialized,
        "device": search_model.device if search_model.initialized else None
    })

@app.route('/extract', methods=['POST'])
def extract_features():
    """Extract features from an image"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'image' field in request body"
            }), 400

        image_base64 = data['image']
        features, error = search_model.extract_image_features(image_base64)
        
        if features is not None:
            return jsonify({
                "success": True,
                "features": features,
                "feature_size": len(features)
            })
        else:
            return jsonify({
                "success": False,
                "error": error or "Failed to process image"
            }), 500

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }), 500

@app.route('/similarity', methods=['POST'])
def calculate_similarity():
    """Calculate similarity between two feature vectors"""
    try:
        data = request.get_json()
        if not data or 'features1' not in data or 'features2' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'features1' or 'features2' in request body"
            }), 400

        features1 = data['features1']
        features2 = data['features2']
        
        similarity, error = search_model.calculate_similarity(features1, features2)
        
        if similarity is not None:
            return jsonify({
                "success": True,
                "similarity": similarity
            })
        else:
            return jsonify({
                "success": False,
                "error": error or "Failed to calculate similarity"
            }), 500

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)