from flask import Flask, request, jsonify
from keras.models import load_model
import os
import sys
import json
import numpy as np
import cv2
import tempfile

app = Flask(__name__)

ROOT_DIR = os.path.dirname(
os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__))))))
sys.path.append(ROOT_DIR)

CONF_FILE = os.path.join(ROOT_DIR, 'settings.json')
MODEL_DIR = os.path.join(ROOT_DIR, 'src/mle_hw5_package/models')


# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)
    
from src.mle_hw5_package.preprocess import preprocess_images

model = None

def get_model(model_name='model_1.keras'):
    """Loads and returns a trained model."""
    global model
    model_path = os.path.join(MODEL_DIR, model_name)  # Adjusted for directory path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}. "
                                "Please train the model first.")
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Failed to load model")
        sys.exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "File not found in request", 400
    file = request.files['file']
    if file.filename == '':
        return "No file provided", 400
    
    file_content = file.read()  # Read file content
    if not file_content:
        return "File is empty", 400

    try:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, 'input.npy')
        image = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image decoding failed")

        np.save(temp_path, np.array([image]))
        processed_images = preprocess_images(temp_path)
        prediction = model.predict(np.array(processed_images))
        return jsonify(prediction.tolist())
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        os.rmdir(temp_dir)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    get_model()
    app.run(host='0.0.0.0', port=1234)
