# Import libraries
"""
This script imports raw data and saves it locally as .csv files.
"""
# Import libraries
import requests
import os
import logging
import sys
import json

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

CONF_FILE = os.path.join(ROOT_DIR, 'settings.json')
# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

DATA_DIR = os.path.join(ROOT_DIR, conf['directories']['data'])
RAW_DATA_DIR = os.path.join(DATA_DIR, conf['directories']['raw_data'])

# Create logger
logger = logging.getLogger(__name__)

# URL for datasets
DATA_URLS = {
    "item_categories": conf['urls']['data'] + conf['files']['item_categories'],
    "items": conf['urls']['data'] + conf['files']['items'],
    "sales_train": conf['urls']['data'] + conf['files']['sales_train'],
    "shops": conf['urls']['data'] + conf['files']['shops'],
}

# Define methods
def setup_directories() -> None:
    """Create required directories if they don't exist."""
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        logger.info(f"Created directory: {RAW_DATA_DIR}")

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


def main() -> None:
    """Main method."""
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    setup_directories()
    item_categories_dest = os.path.join(RAW_DATA_DIR, conf['files']['item_categories'])
    items_dest = os.path.join(RAW_DATA_DIR, conf['files']['items'])
    sales_train_dest = os.path.join(RAW_DATA_DIR, conf['files']['sales_train'])
    shops_dest = os.path.join(RAW_DATA_DIR, conf['files']['shops'])
    # Mapping destinations to data URLs
    dest_to_url_key = {
        item_categories_dest: "item_categories",
        items_dest: "items",
        sales_train_dest: "sales_train",
        shops_dest: "shops"
    }
    # Download each file
    for dest, url_key in dest_to_url_key.items():
        download_file(DATA_URLS[url_key], dest)

# Executing the script
if __name__ == "__main__":
    main()