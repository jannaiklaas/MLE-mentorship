import os
import sys
import json
import time
import logging
import numpy as np
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
ROOT_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)))))
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PACKAGE_DIR)

CONF_FILE = os.path.join(PACKAGE_DIR, 'config/settings.json')

# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

from preprocess import preprocess_images, augment_data
from utils import set_seed

DATA_DIR = os.path.join(ROOT_DIR, conf['directories']['data'])
TRAIN_DIR = os.path.join(DATA_DIR, conf['directories']['train_data'])
TRAIN_IMAGES_PATH = os.path.join(TRAIN_DIR, conf['files']['train_images'])
TRAIN_LABELS_PATH = os.path.join(TRAIN_DIR, conf['files']['train_labels'])
MODEL_DIR = os.path.join(ROOT_DIR, conf['directories']['models'])
RESULTS_DIR = os.path.join(ROOT_DIR, conf['directories']['results'])
BASE_MODEL_URL = conf['urls']['base_model']

def load_data(images_path, train_label_path):
    X = preprocess_images(images_path)
    y = np.load(train_label_path, allow_pickle=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                        test_size=conf['training']['val_size'], 
                                                        random_state=conf['general']['seed_value'])
    logging.info(f"Model will be trained with {X_train.shape[0]} and validated with {X_val.shape[0]} images")
    return X_train, X_val, y_train, y_val

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

def evaluate_model(model, X, y_true, training_time,
                   threshold=conf['training']['decision_threshold']):
    predictions = model.predict(X)
    y_pred = (predictions > threshold).astype(int)
    metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1 Score': f1_score(y_true, y_pred),
            'AUC ROC': roc_auc_score(y_true, predictions),
            'Training duration': training_time
        }
    logging.info("Validation performance metrics:\n"
                     f"Accuracy: {metrics['Accuracy']: .4f}\n"
                     f"Precision: {metrics['Precision']: .4f}\n"
                     f"Recall: {metrics['Accuracy']: .4f}\n"
                     f"F1 score: {metrics['F1 Score']: .4f}\n"
                     f"AUC ROC: {metrics['AUC ROC']: .4f}\n")
    logging.info("Saving performance metrics...")
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
    os.chmod(RESULTS_DIR, 0o775)
    path = os.path.join(RESULTS_DIR, conf['files']['metrics_file'])
    metrics_str = "\n".join([f"{key}: {value}" for key, value in metrics.items()])
    with open(path, "a") as file:
        file.write(metrics_str + "\n\n")
    os.chmod(path, 0o664)
    logging.info(f"Metrics saved to {path}")

def save_model(model, path, 
               model_name) -> None:
    """Saves the trained model to the specified path."""
    logging.info("Saving the model...")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    os.chmod(path, 0o775)
    model_path = os.path.join(path, model_name)
    model.save(model_path)
    os.chmod(model_path, 0o664)  
    logging.info(f"Model saved to {model_path}")

def run_training(images_path=TRAIN_IMAGES_PATH,
                 train_label_path = TRAIN_LABELS_PATH,
                 model_output_path=MODEL_DIR, 
                 model_name=conf['model']['name']+conf['model']['extension']):
    os.umask(0o002)
    set_seed(conf['general']['seed_value'])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    X_train, X_val, y_train, y_val = load_data(images_path, train_label_path)
    train_generator, val_generator = augment_data(X_train, y_train, X_val, y_val)
    base_model, predictions = define_model()
    start_time = time.time()
    model = train_model(base_model, predictions, train_generator, val_generator)
    evaluate_model(model, X_val, y_val, time.time()-start_time)
    save_model(model, model_output_path, model_name)

def main():
    run_training()


# Executing the script
if __name__ == "__main__":
    main()
