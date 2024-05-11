import os
import sys
import cv2
import json
import logging
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

CONF_FILE = os.path.join(ROOT_DIR, 'settings.json')

# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

def preprocess_images(images_path):
    logging.info("Preprocessing images...")
    X_cropped = []
    output_size = tuple(conf['model']['input_shape'][:-1])
    X = np.load(images_path, allow_pickle=True)
    for image in X:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Changing the color from BGR to RGB
        image = cv2.resize(image, output_size)
        image_resized = crop_brain_region(image, output_size)
        X_cropped.append(image_resized)
    X_cropped = np.array(X_cropped)
    return X_cropped / conf['preprocess']['max_binary_value']  # Normalize the images

def crop_brain_region(image, size):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur to smooth the image and reduce noise
    gray = cv2.GaussianBlur(gray, conf['preprocess']['blur_kernel'], 
                            conf['preprocess']['blur_sigma'])
    # Threshold the image to create a binary mask
    thresh = cv2.threshold(gray, conf['preprocess']['threshold_value'], 
                           conf['preprocess']['max_binary_value'], cv2.THRESH_BINARY)[1]
    # Perform morphological operations to remove noise
    thresh = cv2.erode(thresh, None, iterations=conf['preprocess']['erode_iter'])
    thresh = cv2.dilate(thresh, None, iterations=conf['preprocess']['dilate_iter'])
    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assume the brain part of the image has the largest contour
    c = max(contours, key=cv2.contourArea)
    # Get the bounding rectangle of the brain part
    x, y, w, h = cv2.boundingRect(c)
    # Crop the image around the bounding rectangle
    cropped_image = image[y:y + h, x:x + w]
    # Resize the cropped image to the needed size
    resized_image = cv2.resize(cropped_image, size)
    return resized_image  # Return only what's necessary

def augment_data(X_train, y_train, X_val, y_val):
    logging.info(f"Applying data augmentation...")
    train_datagen = ImageDataGenerator(
        rotation_range=conf['augmentation']['rotation_range'],
        horizontal_flip=conf['augmentation']['horizontal_flip'],
        vertical_flip=conf['augmentation']['vertical_flip'],
        width_shift_range=conf['augmentation']['width_shift_range'],
        height_shift_range=conf['augmentation']['height_shift_range'],
        shear_range=conf['augmentation']['shear_range'],
        zoom_range=conf['augmentation']['zoom_range'],
    )
    val_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow(X_train, y_train, batch_size=conf['training']['batch_size'])
    val_generator = val_datagen.flow(X_val, y_val, batch_size=conf['training']['batch_size'])
    return train_generator, val_generator

