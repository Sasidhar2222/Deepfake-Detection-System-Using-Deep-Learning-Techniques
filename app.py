from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the Deepfake Detection Model
model = load_model('deepfake_model.h5')

def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

def detect_deepfake_image(image_path):
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return ("Real" if prediction > 0.5 else "Deepfake", round(confidence * 100, 2))

def detect_deepfake_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_predictions = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_rate = max(frame_count // 30, 1)  # Take frames at equal intervals
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            processed_frame = preprocess_image(frame)
            prediction = model.predict(processed_frame)[0][0]
            frame_predictions.append(prediction)

        frame_idx += 1

    cap.release()

    avg_prediction = np.mean(frame_predictions)
    confidence = avg_prediction if avg_prediction > 0.5 else 1 - avg_prediction
    return ("Deepfake" if avg_prediction > 0.5 else "Real", round(confidence * 100, 2))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        result, confidence = detect_deepfake_image(filepath)
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        result, confidence = detect_deepfake_video(filepath)
    else:
        return "Unsupported file format"

    return render_template('result.html', filename=file.filename, result=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)