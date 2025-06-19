import gradio as gr
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
from threading import Thread
import requests
from werkzeug.serving import make_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class CLIPImageSearchModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialized = False
        self.error_message = None
        logger.info(f"Initializing CLIP model on device: {self.device}")
        
        try:
            logger.info("Loading CLIP model...")
            # Load model directly without custom download path for HF Spaces
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.model.eval()
            self.initialized = True
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            self.initialized = False
            self.error_message = str(e)
            logger.error(f"Failed to initialize CLIP model: {e}")

    def extract_image_features(self, image_base64):
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
search_model = CLIPImageSearchModel()

# Flask API setup
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_initialized": search_model.initialized,
        "device": search_model.device if search_model.initialized else None,
        "error": search_model.error_message if not search_model.initialized else None
    })

@app.route('/extract', methods=['POST'])
def extract_features():
    try:
        if not search_model.initialized:
            return jsonify({
                "success": False,
                "error": f"Model not ready: {search_model.error_message}"
            }), 503

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
    try:
        if not search_model.initialized:
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

# Gradio Interface for testing
def process_image_demo(image):
    if image is None:
        return "Please upload an image"
    
    try:
        # Convert PIL image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        features, error = search_model.extract_image_features(img_base64)
        if features:
            return f"✅ Features extracted successfully!\nFeature vector size: {len(features)}\nFirst 5 values: {features[:5]}"
        else:
            return f"❌ Error: {error}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="CLIP Image Search API") as demo:
    gr.Markdown("# CLIP Image Search API")
    gr.Markdown("This API extracts features from images using CLIP model and calculates similarities.")
    
    with gr.Tab("Test Image Processing"):
        image_input = gr.Image(type="pil", label="Upload Image")
        process_btn = gr.Button("Process Image")
        result_output = gr.Textbox(label="Result", lines=5)
        
        process_btn.click(
            fn=process_image_demo,
            inputs=image_input,
            outputs=result_output
        )
    
    with gr.Tab("API Endpoints"):
        gr.Markdown("""
        ## Available API Endpoints:
        
        ### 1. Health Check
        - **URL**: `/health`
        - **Method**: GET
        - **Response**: Model status and device info
        
        ### 2. Extract Features
        - **URL**: `/extract`
        - **Method**: POST
        - **Body**: `{"image": "base64_encoded_image"}`
        - **Response**: `{"success": true, "features": [...], "feature_size": 512}`
        
        ### 3. Calculate Similarity
        - **URL**: `/similarity`
        - **Method**: POST
        - **Body**: `{"features1": [...], "features2": [...]}`
        - **Response**: `{"success": true, "similarity": 0.85}`
        
        ### Base URL
        Your API will be available at: `https://your-space-name-your-username.hf.space`
        """)

# Run Flask server in a separate thread
def run_flask():
    server = make_server('0.0.0.0', 7860, app, threaded=True)
    server.serve_forever()

if __name__ == "__main__":
    # Start Flask server in background
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Launch Gradio interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False
    )