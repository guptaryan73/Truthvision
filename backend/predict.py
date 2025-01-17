import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Constants
IMG_HEIGHT = 150
IMG_WIDTH = 150
MODEL_PATH = 'deepfake_detection_model.h5'
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}  # Extend as necessary
FRAME_RATE = 1  # Adjust as needed

# Load the trained model and Haar Cascade for face detection
model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Create the Flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper function to check file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
                face_resized = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
                faces.append(face_resized)
        frame_count += 1
    cap.release()
    return faces, None

# Function to predict if faces are real or fake
def predict_faces(faces):
    if not faces:
        return []

    faces_array = np.array(faces) / 255.0  # Normalize the images
    predictions = model.predict(faces_array)
    return predictions

# Route to handle video upload and process
@app.route('/upload_video', methods=['POST'])
def upload_video():
    global count_REAL, count_FAKE  # Declare the counts as global
    count_REAL, count_FAKE = 0, 0

    if 'video' not in request.files:
        print("No video file in request")
        return jsonify({"error": "No video file in request"}), 400
    
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        print("Invalid video file")
        return jsonify({"error": "Invalid video file"}), 400

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    
    print(f"Video uploaded and saved to {video_path}")

    faces, error = extract_faces_from_video(video_path)
    if error:
        print(f"Error during face extraction: {error}")
        return jsonify({"error": error}), 500

    predictions = predict_faces(faces)
    
    for pred in predictions:
        label = 'REAL' if pred[0] >= 0.5 else 'FAKE'
        if label == 'REAL':
            count_REAL += 1
        else:
            count_FAKE += 1

    final_decision = 'REAL' if count_REAL >= count_FAKE else 'FAKE'

    # Log the results to the server console
    print(f"Real Frames: {count_REAL}, Fake Frames: {count_FAKE}")
    print(f"Final Decision: {final_decision}")

    return jsonify({
        "real_frames": count_REAL,
        "fake_frames": count_FAKE,
        "final_decision": final_decision,
    })

if __name__ == '__main__':
    app.run(debug=True)
