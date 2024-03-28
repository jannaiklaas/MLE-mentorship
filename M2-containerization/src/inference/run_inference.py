import os
import sys
import json
import logging
import numpy as np
from keras.models import load_model

# Define directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

CONF_FILE = os.path.join(ROOT_DIR, 'settings.json')

# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

from src.preprocess import preprocess_images
from src.train.train import evaluate_model
from src.utils import set_seed

DATA_DIR = os.path.join(ROOT_DIR, conf['directories']['data'])
INFERENCE_DIR = os.path.join(DATA_DIR, conf['directories']['inference_data'])
INFERENCE_IMAGES_PATH = os.path.join(INFERENCE_DIR, conf['files']['inference_images'])
INFERENCE_LABELS_PATH = os.path.join(INFERENCE_DIR, conf['files']['inference_labels'])
MODEL_DIR = os.path.join(ROOT_DIR, conf['directories']['models'])
RESULTS_DIR = os.path.join(ROOT_DIR, conf['directories']['results'])

def get_model(model_name=conf['model']['name']+conf['model']['extension']):
    """Loads and returns a trained model."""
    logging.info("Loading the pretrained model...")
    model_path = os.path.join(MODEL_DIR, model_name)  # Adjusted for directory path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}. "
                                "Please train the model first.")
    try:
        model = load_model(model_path)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"An error occurred while loading the model from {model_path}: {e}")
        sys.exit(1)

def main():
    """Main method."""
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    set_seed(42)
    X_infer = preprocess_images(INFERENCE_IMAGES_PATH)
    y_infer = np.load(INFERENCE_LABELS_PATH, allow_pickle=True)
    model = get_model()
    evaluate_model(model, X_infer, y_infer, set_name="Inference")

# Executing the script
if __name__ == "__main__":
    main()