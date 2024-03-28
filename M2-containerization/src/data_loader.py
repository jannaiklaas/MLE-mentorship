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
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

CONF_FILE = os.path.join(ROOT_DIR, 'settings.json')
# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

from src.utils import set_seed

DATA_DIR = os.path.join(ROOT_DIR, conf['directories']['data'])
RAW_DATA_DIR = os.path.join(DATA_DIR, conf['directories']['raw_data'])
TRAIN_DIR = os.path.join(DATA_DIR, conf['directories']['train_data'])
INFERENCE_DIR = os.path.join(DATA_DIR, conf['directories']['inference_data'])

from src.utils import set_seed

# Create logger
logger = logging.getLogger(__name__)

# URL for datasets
DATA_URLS = {
    "yes": conf['urls']['data_yes'],
    "no": conf['urls']['data_no']
}

# Define methods
def setup_directories() -> None:
    """Create required directories if they don't exist."""
    for directory in [RAW_DATA_DIR, TRAIN_DIR, INFERENCE_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

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

def download_and_extract_data() -> None:
    """Download and extract data into the 'raw' directory."""
    for label, url in DATA_URLS.items():
        target_dir = os.path.join(RAW_DATA_DIR, label)
        zip_file_path = os.path.join(RAW_DATA_DIR, f"{label}.zip")

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            logger.info(f"Created directory for '{label}' data: {target_dir}")

        # Download and extract ZIP file
        download_file(url, zip_file_path)
        temp_extraction_path = os.path.join(RAW_DATA_DIR, 'temp')
        shutil.unpack_archive(zip_file_path, temp_extraction_path)
        logger.info(f"Extracted '{label}' data into temporary directory")

        # Move from temp extraction path to target directory
        extracted_folder_path = os.path.join(temp_extraction_path, label)
        for filename in os.listdir(extracted_folder_path):
            shutil.move(os.path.join(extracted_folder_path, filename), target_dir)
        logger.info(f"Moved '{label}' data to {target_dir}")

        # Clean up: Remove the ZIP file and temporary extraction directory
        os.remove(zip_file_path)
        shutil.rmtree(temp_extraction_path)
        logger.info(f"Cleaned up temporary files for '{label}'")


def load_split_data() -> None:
    """Load data from 'raw' directory, split into train and inference, and save."""
    set_seed(42)
    path_yes = os.path.join(RAW_DATA_DIR, 'yes', '*')
    path_no = os.path.join(RAW_DATA_DIR, 'no', '*')
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
    X_train, X_inference, y_train, y_inference = train_test_split(X_shuffled, 
                                                                  y_shuffled, 
                                                                  test_size=conf['inference']['infer_size'], 
                                                                  random_state=conf['general']['seed_value'])
    logger.info(f"Training set contains {X_train.shape[0]} images")
    logger.info(f"Inference set contains {X_inference.shape[0]} images")

    # Save
    for name, dataset, directory in [("train", (X_train, y_train), TRAIN_DIR), ("inference", (X_inference, y_inference), INFERENCE_DIR)]:
        np.save(os.path.join(directory, f"{name}_images.npy"), dataset[0])
        np.save(os.path.join(directory, f"{name}_labels.npy"), dataset[1])
        logger.info(f"Saved {name} data and labels to {directory}")

def main() -> None:
    """Main method."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    setup_directories()
    download_and_extract_data()
    load_split_data()

# Executing the script
if __name__ == "__main__":
    main()
