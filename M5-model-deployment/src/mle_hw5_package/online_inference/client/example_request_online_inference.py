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
PACKAGE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))

sys.path.append(PACKAGE_DIR)
CONF_FILE = os.path.join(PACKAGE_DIR, 'config/settings.json')

# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

DATA_DIR = os.path.join(ROOT_DIR, conf['directories']['data'])
DEFAULT_EXAMPLE = conf['flask']['default_example']

# Decide the API URL based on the environment variable
if os.getenv("DOCKER_ENV") == "TRUE":
    KERAS_REST_API_URL = conf['flask']['docker_server']
else:
    KERAS_REST_API_URL = conf['flask']['local_server']

# Setup command-line argument parsing
parser = argparse.ArgumentParser(description="Send an image to the Keras REST API for prediction")
parser.add_argument("-i", "--image", type=str, default=DEFAULT_EXAMPLE,
                    help="absolute path to the image file")
args = parser.parse_args()

if args.image==DEFAULT_EXAMPLE: 
    IMAGE_PATH = os.getenv("IMAGE_PATH", DEFAULT_EXAMPLE)
    if IMAGE_PATH=="" or IMAGE_PATH==None:
        IMAGE_PATH = os.path.join(DATA_DIR, DEFAULT_EXAMPLE)
    else: 
        IMAGE_PATH = os.path.join(DATA_DIR, IMAGE_PATH)
else:
    IMAGE_PATH = args.image

def wait_for_server(url, image_path, retries=conf['flask']['retries'], delay=conf['flask']['delay']):
    """Attempts to send the request repeatedly until success or max retries reached."""
    for i in range(retries):
        response = send_request(url, image_path)
        if response and response.status_code == conf['flask']['success']:
            return response
        else:
            print(f"Attempt {i+1}/{retries} failed, waiting {delay} seconds...")
            time.sleep(delay)
    return response

# Load the input image and construct the payload for the request
def send_request(url, image_path):
    with open(image_path, "rb") as image_file:
        payload = {"file": (os.path.basename(image_path), image_file, 'image/jpeg/png/jpg')}
        try:
            response = requests.post(url, files=payload)
            return response
        except requests.ConnectionError as e:
            print(f"Server not ready, connection error: {e}")
        except requests.RequestException as e:
            print(f"Request failed: {e}")
        return None

response = wait_for_server(KERAS_REST_API_URL, IMAGE_PATH)

if response and response.status_code == conf['flask']['success']:
    result = response.json()[0][0]
    diagnosis = "positive for brain tumor" if result > conf['training']['decision_threshold'] else "negative for brain tumor"
    print(f"Probability: {result * 100:.2f}%. Diagnosis: {diagnosis}.")
else:
    print("Request failed.")
    if response:
        print(f"Status Code: {response.status_code}. Response: {response.text}")