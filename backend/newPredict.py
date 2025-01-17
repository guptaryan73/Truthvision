import os
import cv2
import numpy as np
import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Constants
IMG_HEIGHT = 299
IMG_WIDTH = 299
MODEL_PATH = 'model.pt'
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
FRAME_RATE = 1

# Create the Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, methods=["GET", "POST", "OPTIONS"])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the face cascade
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Function to recreate the XceptionNet model architecture
def create_model():
    model = timm.create_model('xception', pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification
    return model

# Function to load model weights
def load_model(model_path):
    model = create_model()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()  # Set model to evaluation mode
    return model

# Load the model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Preprocessing function
def preprocess_frame(image, input_size=(IMG_HEIGHT, IMG_WIDTH)):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to extract faces from video
def extract_faces_from_video(video_path):
    faces = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return faces, "Error opening video file."
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % FRAME_RATE == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in detected_faces:
                face = frame[y:y+h, x:x+w]
                faces.append(face)
        frame_count += 1
    cap.release()
    return faces, None

# Predict if faces are real or fake
def predict_faces(faces):
    predictions = []
    with torch.no_grad():
        for face in faces:
            input_tensor = preprocess_frame(face)
            output = model(input_tensor)
            probability = torch.sigmoid(output)
            prediction = (probability > 0.5).float().item()  # Binary prediction
            predictions.append(prediction)
    return predictions

# Route to handle video upload and processing
@app.route('/upload_video', methods=['POST'])
def upload_video():
    global count_REAL, count_FAKE
    count_REAL, count_FAKE = 0, 0

    if 'video' not in request.files:
        return jsonify({"error": "No video file in request"}), 400
    
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid video file"}), 400

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)

    faces, error = extract_faces_from_video(video_path)
    if error:
        return jsonify({"error": error}), 500

    predictions = predict_faces(faces)

    for pred in predictions:
        label = 'REAL' if pred >= 0.5 else 'FAKE'
        if label == 'REAL':
            count_REAL += 1
        else:
            count_FAKE += 1

    final_decision = 'REAL' if count_REAL >= count_FAKE else 'FAKE'

    return jsonify({
        "real_frames": count_REAL,
        "fake_frames": count_FAKE,
        "final_decision": final_decision,
    })

# Helper function to check file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
