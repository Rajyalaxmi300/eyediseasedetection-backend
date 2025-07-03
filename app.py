import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
import pymongo
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.h5")
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

# Define class names
CLASS_NAMES = ['Bulging_Eyes', 'Cataracts', 'Glaucoma', 'Uveitis']

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "eyediseaseDB"
IMAGES_COLLECTION = "images"
PREDICTIONS_COLLECTION = "predictions"

# Connect to MongoDB
try:
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    images_collection = db[IMAGES_COLLECTION]
    predictions_collection = db[PREDICTIONS_COLLECTION]
    print("Connected to MongoDB successfully")
except Exception as e:
    print(f"Error connecting to MongoDB: {str(e)}")

# Load the trained model with error handling
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print(f"Please make sure the model file exists at: {MODEL_PATH}")
    model = None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if needed
        if len(image.shape) == 2:  # If grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # If RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:  # If BGR (OpenCV default)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, (224, 224))
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
        
    except Exception as e:
        print(f"Error in image preprocessing: {str(e)}")
        return None

def normalize_predictions(predictions):
    """Normalize predictions to valid probabilities"""
    # Apply softmax
    exp_preds = np.exp(predictions - np.max(predictions))  # Subtract max for numerical stability
    probabilities = exp_preds / np.sum(exp_preds)
    return probabilities

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Model not loaded. Please train the model first.",
            "success": False
        }), 500

    try:
        # Check if image file is present in request
        if "image" not in request.files:
            return jsonify({
                "error": "No image file provided",
                "success": False
            }), 400

        # Get the image from the request
        file = request.files["image"]
        
        # Check if a file was actually selected
        if file.filename == "":
            return jsonify({
                "error": "No file selected",
                "success": False
            }), 400

        # Check file extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                "error": f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}",
                "success": False
            }), 400

        # Read and preprocess the image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({
                "error": "Failed to read image",
                "success": False
            }), 400

        # Preprocess image
        processed_img = preprocess_image(img)
        if processed_img is None:
            return jsonify({
                "error": "Failed to preprocess image",
                "success": False
            }), 500
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)[0]
        
        # Print raw predictions for debugging
        print("Raw model output:", predictions)
        
        # Ensure predictions are in the right format
        predictions = np.array(predictions)
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(1, -1)
        
        # Apply softmax to get probabilities
        exp_preds = np.exp(predictions - np.max(predictions))
        probabilities = exp_preds / np.sum(exp_preds)
        probabilities = probabilities.flatten()  # Ensure 1D array
        
        print("Probabilities after softmax:", probabilities)
        print("Sum of probabilities:", np.sum(probabilities))
        
        # Get all predictions with confidence scores
        all_predictions = []
        for i, prob in enumerate(probabilities):
            confidence = float(prob * 100)  # Convert to percentage
            all_predictions.append({
                "class": CLASS_NAMES[i],
                "confidence": round(confidence, 1)
            })
        
        # Sort by confidence
        all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        top_prediction = all_predictions[0]
        
        print("All predictions:", all_predictions)
        print("Top prediction:", top_prediction)
        
        # Return the top prediction regardless of confidence
        return jsonify({
            "class": top_prediction["class"],
            "confidence": top_prediction["confidence"],
            "all_predictions": all_predictions,
            "success": True
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full error traceback
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route("/images", methods=["GET"])
def get_images():
    """Fetch stored image metadata from MongoDB"""
    try:
        images = list(images_collection.find({}, {"_id": 0}))
        return jsonify({
            "success": True,
            "count": len(images),
            "images": images
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Eye Disease Detection API is running",
        "status": "active",
        "model_loaded": model is not None,
        "classes": CLASS_NAMES,
        "total_classes": len(CLASS_NAMES)
    })

if __name__ == "__main__":
    app.run(debug=True)
