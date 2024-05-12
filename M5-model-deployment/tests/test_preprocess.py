import pytest
import numpy as np
import cv2
import os
import json
import sys
from unittest.mock import patch

# Load the config settings required for testing
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
PACKAGE_DIR = os.path.join(ROOT_DIR, 'src', 'mle_hw5_package')

CONF_FILE = os.path.join(PACKAGE_DIR, 'config/settings.json')
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

from mle_hw5_package.preprocess import crop_brain_region, preprocess_images, augment_data

def mock_threshold(image, thresh_value, max_value, type):
    # Create a clear binary image where the bright rectangle will definitely be above the threshold
    binary_image = np.zeros_like(image, dtype=np.uint8)
    binary_image[image > thresh_value] = max_value
    return thresh_value, binary_image

def mock_gaussian_blur(image, kernel_size, sigma):
    # Simply return the input image as Gaussian Blur is not critical for the contour logic in the test
    return image

def mock_cvtColor(image, flag):
    if flag == cv2.COLOR_RGB2GRAY:
        return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    elif flag == cv2.COLOR_BGR2RGB:
        return image[:, :, ::-1]
    return image

def mock_resize(image, size):
    return np.zeros((size[0], size[1], image.shape[2]), dtype=image.dtype)

@pytest.fixture
def test_image():
    # Create an image with a bright rectangle to ensure it creates a contour
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)  # White rectangle
    return img

def test_crop_brain_region(test_image):
    # Use the parameters from the configuration
    size = tuple(conf['model']['input_shape'][:-1])
    cropped_image = crop_brain_region(test_image, size)
    assert cropped_image.shape == (size[0], size[1], 3)
    # The entire image should not be black (assuming there's some part of the brain in the image)
    assert np.any(cropped_image != 0)

@patch('cv2.cvtColor', side_effect=mock_cvtColor)
@patch('cv2.resize', side_effect=mock_resize)
@patch('cv2.threshold', side_effect=mock_threshold)
@patch('cv2.GaussianBlur', side_effect=mock_gaussian_blur)
def test_preprocess_images(mock_gaussian_blur, mock_threshold, mock_resize, mock_cvtcolor, test_image):
    images_array = [test_image, test_image]
    preprocessed_images = preprocess_images(images_array=images_array)
    print("Preprocessed Image Shape:", preprocessed_images.shape)
    print("Max pixel value in preprocessed:", np.max(preprocessed_images))
    assert preprocessed_images.shape == (2, conf['model']['input_shape'][0], conf['model']['input_shape'][1], 3)
    assert np.all(preprocessed_images <= 1.0) and np.all(preprocessed_images >= 0)

def test_augment_data():
    X_train = np.random.rand(10, conf['model']['input_shape'][0], conf['model']['input_shape'][1], 3)
    y_train = np.random.randint(0, 2, size=(10,))
    X_val = np.random.rand(5, conf['model']['input_shape'][0], conf['model']['input_shape'][1], 3)
    y_val = np.random.randint(0, 2, size=(5,))
    
    train_gen, val_gen = augment_data(X_train, y_train, X_val, y_val)
    assert train_gen.batch_size == conf['training']['batch_size']
    assert val_gen.batch_size == conf['training']['batch_size']
    assert train_gen.n == len(X_train)
    assert val_gen.n == len(X_val)
