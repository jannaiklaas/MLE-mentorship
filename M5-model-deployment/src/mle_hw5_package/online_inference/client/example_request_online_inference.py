# import the necessary packages
import requests
import os
import sys
import json
import time
import argparse

# initialize the Keras REST API endpoint URL along with the input
# image path
ROOT_DIR = os.path.dirname(
os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__))))))
sys.path.append(ROOT_DIR)

CONF_FILE = os.path.join(ROOT_DIR, 'settings.json')

# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

DATA_DIR = os.path.join(ROOT_DIR, conf['directories']['data'])
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
DEFAULT_EXAMPLE = "yes/Y19.jpg"

# Decide the API URL based on the environment variable
if os.getenv("DOCKER_ENV") == "TRUE":
    KERAS_REST_API_URL = "http://server:1234/predict"
else:
    KERAS_REST_API_URL = "http://localhost:1234/predict"

# Setup command-line argument parsing
parser = argparse.ArgumentParser(description="Send an image to the Keras REST API for prediction")
parser.add_argument("-i", "--image", type=str, default=DEFAULT_EXAMPLE,
                    help="path to the image file relative to the 'raw' data directory")
args = parser.parse_args()

if args.image==DEFAULT_EXAMPLE: 
    IMAGE_PATH = os.getenv("IMAGE_PATH", DEFAULT_EXAMPLE)
    IMAGE_PATH = os.path.join(RAW_DATA_DIR, IMAGE_PATH)
else:
    IMAGE_PATH = args.image

def wait_for_server(url, image_path, retries=5, delay=10):
    """Attempts to send the request repeatedly until success or max retries reached."""
    for i in range(retries):
        response = send_request(url, image_path)
        if response and response.status_code == 200:
            return response
        else:
            print(f"Attempt {i+1}/{retries} failed, waiting {delay} seconds...")
            time.sleep(delay)
    return response

# Load the input image and construct the payload for the request
def send_request(url, image_path):
    with open(image_path, "rb") as image_file:
        payload = {"file": (os.path.basename(image_path), image_file, 'image/jpeg')}
        try:
            response = requests.post(url, files=payload)
            return response
        except requests.ConnectionError as e:
            print(f"Server not ready, connection error: {e}")
        except requests.RequestException as e:
            print(f"Request failed: {e}")
        return None

response = wait_for_server(KERAS_REST_API_URL, IMAGE_PATH)

if response and response.status_code == 200:
    result = response.json()[0][0]
    diagnosis = "positive for brain tumor" if result > 0.5 else "negative for brain tumor"
    print(f"Probability: {result * 100:.2f}%. Diagnosis: {diagnosis}.")
else:
    print("Request failed.")
    if response:
        print(f"Status Code: {response.status_code}. Response: {response.text}")