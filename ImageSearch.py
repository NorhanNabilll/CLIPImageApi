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
import logging
from flask_cors import CORS
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

class CLIPImageSearchModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialized = False
        self.error_message = None
        self.model = None
        self.preprocess = None
        logger.info(f"Initializing CLIP model on device: {self.device}")
        
        # Initialize model in background thread
        self.init_thread = threading.Thread(target=self._initialize_model)
        self.init_thread.start()

    def _initialize_model(self):
        """Initialize model in background thread"""
        try:
            logger.info("Starting CLIP model download/initialization...")
            start_time = time.time()
            
            # Load model - HF Spaces handles caching automatically
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.model.eval()
            self.initialized = True
            
            elapsed_time = time.time() - start_time
            logger.info(f"CLIP model loaded successfully in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            self.initialized = False
            self.error_message = str(e)
            logger.error(f"Failed to initialize CLIP model: {e}")

    def wait_for_initialization(self, timeout=300):
        """Wait for model initialization with timeout"""
        if self.init_thread.is_alive():
            self.init_thread.join(timeout=timeout)
        return self.initialized

    def extract_image_features(self, image_base64):
        if not self.initialized:
            # Wait a bit more if still initializing
            if self.init_thread.is_alive():
                logger.info("Model still initializing, waiting...")
                self.init_thread.join(timeout=30)
            
            if not self.initialized:
                return None, f"Model not initialized: {self.error_message}"
        
        try:
            logger.info("Processing image for feature extraction")
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten().tolist(), None
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None, f"Error processing image: {str(e)}"

    def calculate_similarity(self, features1, features2):
        try:
            features1 = np.array(features1)
            features2 = np.array(features2)
            similarity = np.dot(features1, features2)
            return float(similarity), None
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return None, f"Error calculating similarity: {str(e)}"

# Initialize the model globally
logger.info("Starting model initialization...")
search_model = CLIPImageSearchModel()

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        "message": "CLIP Image Search API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Check API and model status",
            "/wait-ready": "GET - Wait for model to be ready",
            "/extract": "POST - Extract features from image",
            "/similarity": "POST - Calculate similarity between features"
        },
        "model_status": {
            "initialized": search_model.initialized,
            "device": search_model.device,
            "error": search_model.error_message if not search_model.initialized else None
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_initialized": search_model.initialized,
        "model_initializing": search_model.init_thread.is_alive() if search_model.init_thread else False,
        "device": search_model.device,
        "error": search_model.error_message if not search_model.initialized else None
    })

@app.route('/wait-ready', methods=['GET'])
def wait_ready():
    """Endpoint to wait for model to be ready"""
    timeout = int(request.args.get('timeout', 300))
    ready = search_model.wait_for_initialization(timeout)
    
    return jsonify({
        "ready": ready,
        "model_initialized": search_model.initialized,
        "error": search_model.error_message if not ready else None
    })

@app.route('/extract', methods=['POST'])
def extract_features():
    """Extract features from an image"""
    try:
        # Check if model is initialized
        if not search_model.initialized:
            # Try waiting for initialization
            logger.info("Model not ready, attempting to wait for initialization...")
            ready = search_model.wait_for_initialization(timeout=60)
            
            if not ready:
                return jsonify({
                    "success": False,
                    "error": f"Model not ready: {search_model.error_message}",
                    "model_initializing": search_model.init_thread.is_alive() if search_model.init_thread else False
                }), 503  # Service Unavailable
        
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
        logger.error(f"Unexpected error in extract_features: {e}")
        return jsonify({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }), 500

@app.route('/similarity', methods=['POST'])
def calculate_similarity():
    """Calculate similarity between two feature vectors"""
    try:
        # Check if model is initialized
        if not search_model.initialized:
            ready = search_model.wait_for_initialization(timeout=60)
            if not ready:
                return jsonify({
                    "success": False,
                    "error": f"Model not ready: {search_model.error_message}"
                }), 503
        
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
        logger.error(f"Unexpected error in calculate_similarity: {e}")
        return jsonify({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Starting Flask app on port {port}")
    
    # Give model some time to initialize before serving requests
    logger.info("Waiting for initial model setup...")
    search_model.wait_for_initialization(timeout=30)
    
    app.run(host="0.0.0.0", port=port, debug=False)