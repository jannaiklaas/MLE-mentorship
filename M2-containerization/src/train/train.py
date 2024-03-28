import os
import sys
import json
import logging
import numpy as np
from datetime import datetime
from keras.models import Model
from keras.utils import get_file
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, roc_auc_score
from keras.applications import VGG16
from keras import regularizers


# Define directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

CONF_FILE = os.path.join(ROOT_DIR, 'settings.json')

# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

from src.preprocess import preprocess_images, augment_data
from src.utils import set_seed

DATA_DIR = os.path.join(ROOT_DIR, conf['directories']['data'])
TRAIN_DIR = os.path.join(DATA_DIR, conf['directories']['train_data'])
TRAIN_IMAGES_PATH = os.path.join(TRAIN_DIR, conf['files']['train_images'])
TRAIN_LABELS_PATH = os.path.join(TRAIN_DIR, conf['files']['train_labels'])
MODEL_DIR = os.path.join(ROOT_DIR, conf['directories']['models'])
RESULTS_DIR = os.path.join(ROOT_DIR, conf['directories']['results'])
BASE_MODEL_URL = conf['urls']['base_model']

def define_model():
    logging.info("Retrieving the base model...")
    base_model = VGG16(weights=None, include_top=False, 
                       input_shape=conf['model']['input_shape'])
    weights_path = get_file(origin=BASE_MODEL_URL)
    base_model.load_weights(weights_path)
    base_model.layers[0].trainable = False
    logging.info("Modifying the base model...")
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(conf['model']['input_shape'][0], activation='relu', 
              kernel_regularizer=regularizers.l2(conf['model']['regularization_rate']))(x)
    x = Dropout(conf['model']['dropout_rate'])(x)
    predictions = Dense(1, activation='sigmoid')(x)
    logging.info("Model architecture finalized.")
    return base_model, predictions

def train_model(base_model, predictions, train_generator, val_generator):
    logging.info("Training the model...")
    set_seed(conf['general']['seed_value'])
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(learning_rate=conf['model']['learning_rate']), 
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=conf['training']['early_stop_patience'], 
                                   restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                  factor=conf['training']['reduce_lr_factor'], 
                                  patience=conf['training']['reduce_lr_patience'], 
                                  min_lr=conf['training']['min_lr'])
    history = model.fit(
        train_generator,
        epochs=conf['training']['epochs'],
        validation_data=val_generator,
        callbacks=[early_stopping, reduce_lr]
    )
    logging.info("Training complete.")
    return model

def evaluate_model(model, X, y_true, set_name, 
                   threshold=conf['training']['decision_threshold'], 
                   model_name=conf['model']['name']):
    logging.info("Evaluating the model...")
    predictions = model.predict(X)
    y_pred = (predictions > threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, predictions)
    metrics = {
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Model': model_name,
            'Status': set_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC ROC': auc_roc
        }
    logging.info("Performance metrics:\n"
                     f"Model name: {model_name}\n"
                     f"Status: {set_name}\n"
                     f"Accuracy: {accuracy: .4f}\n"
                     f"Precision: {precision: .4f}\n"
                     f"Recall: {recall: .4f}\n"
                     f"F1 score: {f1: .4f}\n"
                     f"AUC ROC: {auc_roc: .4f}\n")
    logging.info("Saving performance metrics...")
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    path = os.path.join(RESULTS_DIR, conf['files']['metrics_file'])
    metrics_str = "\n".join([f"{key}: {value}" for key, value in metrics.items()])
    with open(path, "a") as file:
        file.write(metrics_str + "\n\n")
    logging.info(f"Metrics saved to {path}")

def save_model(model, path, model_name=conf['model']['name']+conf['model']['extension']) -> None:
    """Saves the trained model to the specified path."""
    logging.info("Saving the model...")
    if not os.path.exists(path):
        os.makedirs(path)
    model_path = os.path.join(path, model_name)
    model.save(model_path)  
    logging.info(f"Model saved to {model_path}")

def main():
    set_seed(conf['general']['seed_value'])
    """Main method."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    X = preprocess_images(TRAIN_IMAGES_PATH)
    y = np.load(TRAIN_LABELS_PATH, allow_pickle=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                      test_size=conf['training']['val_size'], 
                                                      random_state=conf['general']['seed_value'])
    logging.info(f"Model will be trained with {X_train.shape[0]} and validated with {X_val.shape[0]} images")
    train_generator, val_generator = augment_data(X_train, y_train, X_val, y_val)
    base_model, predictions = define_model()
    model = train_model(base_model, predictions, train_generator, val_generator)
    evaluate_model(model, X_val, y_val, set_name="Validation")
    save_model(model, MODEL_DIR)


# Executing the script
if __name__ == "__main__":
    main()
