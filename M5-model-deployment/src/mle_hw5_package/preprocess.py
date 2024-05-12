import os
import sys
import cv2
import json
import logging
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

CONF_FILE = os.path.join(ROOT_DIR, 'config/settings.json')

# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

def preprocess_images(images_path=None, images_array=None):
    logging.info("Preprocessing images...")
    X_cropped = []
    output_size = tuple(conf['model']['input_shape'][:-1])
    X = np.load(images_path, allow_pickle=True) if images_path!=None else images_array
    for image in X:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Changing the color from BGR to RGB
        image = cv2.resize(image, output_size)
        image_resized = crop_brain_region(image, output_size)
        X_cropped.append(image_resized)
    X_cropped = np.array(X_cropped)
    return X_cropped / conf['preprocess']['max_binary_value']  # Normalize the images

def crop_brain_region(image, size):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, conf['preprocess']['blur_kernel'], conf['preprocess']['blur_sigma'])
    _, thresh = cv2.threshold(gray, conf['preprocess']['threshold_value'], conf['preprocess']['max_binary_value'], cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=conf['preprocess']['erode_iter'])
    thresh = cv2.dilate(thresh, None, iterations=conf['preprocess']['dilate_iter'])
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cropped_image = image[y:y+h, x:x+w]
        return cv2.resize(cropped_image, size)
    else:
        return cv2.resize(image, size)
 

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

