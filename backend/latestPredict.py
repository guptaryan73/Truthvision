import os
import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Import Flask-CORS
import time  # Import time module

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, methods=["GET", "POST", "OPTIONS"])

# Folder to store uploaded videos
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for video files
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Function to recreate the XceptionNet model architecture
def create_model():
    model = timm.create_model('xception', pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Assuming binary classification
    return model

# Function to load model weights properly
def load_model(model_path):
    print("Loading model from", model_path)
    model = create_model()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully!")
    return model

# Preprocessing function for each frame
def preprocess_frame(image, input_size=(299, 299)):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Predict if a video is a deepfake or real
def predict_deepfake(model, frames):
    print("Analyzing frames...")
    predictions = []
    with torch.no_grad():
        for idx, frame in enumerate(frames):
            input_tensor = preprocess_frame(frame)
            output = model(input_tensor)
            probability = torch.sigmoid(output)
            prediction = (probability > 0.5).float().item()
            predictions.append(prediction)
            
            if (idx + 1) % 10 == 0 or idx == len(frames) - 1:
                print(f"Processed {idx + 1}/{len(frames)} frames...")
    return predictions

# Function to extract frames from video
def extract_frames(input_path, frame_skip=20):
    print(f"Extracting frames from {input_path}...")
    cap = cv2.VideoCapture(input_path)
    frames = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_skip == 0:
            frames.append(frame)
        frame_id += 1

    cap.release()
    print(f"Extracted {len(frames)} frames.")
    return frames

# Utility function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the model (specify the correct model path)
model_path = 'model.pt'  # Replace with your model path
model = load_model(model_path)

# Route to handle video uploads and predictions
@app.route('/upload_video', methods=['POST'])
def upload_video():
    # Check if a file is present in the request
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']

    # Check if the file has a valid extension
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        print(f"Video '{filename}' uploaded successfully.")

        # Extract frames from the video
        frames = extract_frames(video_path)

        # Start timing
        start_time = time.time()

        # Get deepfake predictions
        predictions = predict_deepfake(model, frames)
        deepfake_count = sum(predictions)  # Count frames predicted as deepfake
        real_count = len(predictions) - deepfake_count  # Calculate real frames

        # Calculate time taken for prediction
        elapsed_time = time.time() - start_time

        # Determine if the video is a deepfake
        if deepfake_count > len(predictions) / 2:
            result = "The video is likely a deepfake."
        else:
            result = "The video is likely real."

        # Log time and result to console
        print(f"Analysis complete. Result: {result}")
        print(f"Time taken for detection: {elapsed_time:.2f} seconds")
        print(f"Number of real frames: {real_count}, Fake frames: {deepfake_count}")

        # Return the result, elapsed time, and frame counts in the response
        return jsonify({
            'result': result,
            'time_taken': f"{elapsed_time:.2f} seconds",
            'real_frames': real_count,
            'fake_frames': deepfake_count
        }), 200
    else:
        print("Invalid file format.")
        return jsonify({'error': 'Invalid file format'}), 400

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
