# Import libraries
import requests
import os
import json
import logging
import sys
import shutil
import cv2
import glob
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Define directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PACKAGE_DIR)

from utils import set_seed

CONF_FILE = os.path.join(PACKAGE_DIR, 'config/settings.json')
# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

DATA_DIR = os.path.join(ROOT_DIR, conf['directories']['data'])

# Create logger
logger = logging.getLogger(__name__)

# URL for datasets
DATA_URLS = {
    "yes": conf['urls']['data_yes'],
    "no": conf['urls']['data_no']
}

# Define methods
def setup_directories(data_output_path) -> None:
    """Create required directories if they don't exist."""
    raw_data_dir = os.path.join(data_output_path, conf['directories']['raw_data'])
    train_dir = os.path.join(data_output_path, conf['directories']['train_data'])
    inference_dir = os.path.join(data_output_path, conf['directories']['inference_data'])
    for directory in [raw_data_dir, train_dir, inference_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    return raw_data_dir, train_dir, inference_dir

def download_file(url: str, filename: str) -> None:
    """Download file from a given URL and save it to a specified filename."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=conf['general']['chunk_size']):
                file.write(chunk)
        logger.info(f"Downloaded and saved file: {filename}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        sys.exit(1)

def download_data(raw_data_dir) -> None:
    """Download and extract data into the 'raw' directory."""
    for label, url in DATA_URLS.items():
        target_dir = os.path.join(raw_data_dir, label)
        zip_file_path = os.path.join(raw_data_dir, f"{label}.zip")

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            logger.info(f"Created directory for '{label}' data: {target_dir}")

        # Download and extract ZIP file
        download_file(url, zip_file_path)
        temp_extraction_path = os.path.join(raw_data_dir, 'temp')
        shutil.unpack_archive(zip_file_path, temp_extraction_path)
        logger.info(f"Extracted '{label}' data into temporary directory")

        # Move from temp extraction path to target directory
        extracted_folder_path = os.path.join(temp_extraction_path, label)
        for filename in os.listdir(extracted_folder_path):
            src_file = os.path.join(extracted_folder_path, filename)
            dest_file = os.path.join(target_dir, filename)
            if os.path.exists(dest_file):
                os.remove(dest_file)  # Remove if exists to avoid error
            shutil.move(src_file, dest_file)
        logger.info(f"Moved '{label}' data to {target_dir}")

        # Clean up: Remove the ZIP file and temporary extraction directory
        os.remove(zip_file_path)
        shutil.rmtree(temp_extraction_path)
        logger.info(f"Cleaned up temporary files for '{label}'")

def split_data(raw_data_dir, train_dir, inference_dir) -> None:
    """Load data from 'raw' directory, split into train and inference, and save."""
    set_seed(42)
    path_yes = os.path.join(raw_data_dir, 'yes', '*')
    path_no = os.path.join(raw_data_dir, 'no', '*')
    tumor = []
    no_tumor = []

    # Read and collect data
    for file in glob.iglob(path_yes):
        img = cv2.imread(file)
        tumor.append((img, 1))

    for file in glob.iglob(path_no):
        img = cv2.imread(file)
        no_tumor.append((img, 0))

    # Shuffle and split
    all_data = tumor + no_tumor
    X = np.array([item[0] for item in all_data])
    y = np.array([item[1] for item in all_data])
    X_shuffled, y_shuffled = shuffle(X, y, random_state=conf['general']['seed_value'])
    X_train, X_inference, y_train, y_inference = train_test_split(X_shuffled, y_shuffled,
                                            test_size=conf['inference']['infer_size'], 
                                            random_state=conf['general']['seed_value'])
    
    logger.info(f"Training set contains {len(X_train)} images")
    logger.info(f"Inference set contains {len(X_inference)} images")

    # Save training data to .npy files
    np.save(os.path.join(train_dir, "train_images.npy"), X_train)
    np.save(os.path.join(train_dir, "train_labels.npy"), y_train)
    logger.info(f"Saved training data and labels to {train_dir}")
    os.makedirs(inference_dir, exist_ok=True)

    for i, img in enumerate(X_inference):
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(inference_dir, filename), img)

    logger.info(f"Saved inference images to {inference_dir}")

def download_split_data(data_output_path=DATA_DIR):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    raw_data_dir, train_dir, inference_dir = setup_directories(data_output_path)
    download_data(raw_data_dir)
    split_data(raw_data_dir, train_dir, inference_dir)


def main() -> None:
    """Main method."""
    download_split_data()

# Executing the script
if __name__ == "__main__":
    main()