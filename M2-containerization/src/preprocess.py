import cv2
import logging
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_images(images_path):
    logging.info("Preprocessing images...")
    X_cropped = []
    output_size = (128, 128)
    X = np.load(images_path, allow_pickle=True)
    for image in X:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Changing the color from BGR to RGB
        image = cv2.resize(image, (128, 128))
        image_resized = crop_brain_region(image, output_size)
        X_cropped.append(image_resized)
    X_cropped = np.array(X_cropped)
    return X_cropped / 255  # Normalize the images

def crop_brain_region(image, size):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur to smooth the image and reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold the image to create a binary mask
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    # Perform morphological operations to remove noise
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
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
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
    )
    val_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=32)
    return train_generator, val_generator
