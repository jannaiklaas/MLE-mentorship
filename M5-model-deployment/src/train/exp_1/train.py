import os
import sys
import json
import time
import mlflow
import logging
import hashlib
from datetime import datetime
import numpy as np
import platform
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, roc_auc_score
from mlflow.models import infer_signature

# Define directories
ROOT_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

CONF_FILE = os.path.join(ROOT_DIR, 'settings.json')

# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

from src.mle_hw5_package.preprocess import preprocess_images, augment_data
from src.mle_hw5_package.utils import set_seed

DATA_DIR = os.path.join(ROOT_DIR, conf['directories']['data'])
TRAIN_DIR = os.path.join(DATA_DIR, conf['directories']['train_data'])
TRAIN_IMAGES_PATH = os.path.join(TRAIN_DIR, conf['files']['train_images'])
TRAIN_LABELS_PATH = os.path.join(TRAIN_DIR, conf['files']['train_labels'])
MODEL_DIR = os.path.join(ROOT_DIR, conf['directories']['models'])
RESULTS_DIR = os.path.join(ROOT_DIR, conf['directories']['results'])
BASE_MODEL_URL = conf['urls']['base_model']

TRAIN_FILE_PATH =  os.path.join(ROOT_DIR, conf['exp_1']['script_file_path'])
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', conf['general']['fallback_uri'])
MLFLOW_EXPERIMENT_NAME = conf['general']['experiment_name']

def file_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    return file_hash

def load_data():
    X = preprocess_images(TRAIN_IMAGES_PATH)
    y = np.load(TRAIN_LABELS_PATH, allow_pickle=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                        test_size=conf['training']['val_size'], 
                                                        random_state=conf['general']['seed_value'])
    logging.info(f"Model will be trained with {X_train.shape[0]} and validated with {X_val.shape[0]} images")

    mlflow.log_param("train_images_shape", str(X_train.shape))
    mlflow.log_param("val_images_shape", str(X_val.shape))

    return X_train, X_val, y_train, y_val

def define_model():
    logging.info("Defining the model...")
    model = Sequential()
    activation = conf['model']['activation']
    input_shape = conf['model']['input_shape']
    kernel_size = conf['model']['kernel_size']
    pool_size = conf['model']['pool_size']
    input_length = input_shape[0]
    model.add(Conv2D(int(input_length/4), kernel_size=kernel_size, activation=activation, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Conv2D(int(input_length/2), kernel_size=kernel_size, activation=activation))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Conv2D(input_length, kernel_size=kernel_size, activation=activation))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dropout(conf['model']['dropout_rate']))
    model.add(Dense(input_length, activation=activation))
    model.add(Dropout(conf['model']['dropout_rate']))
    model.add(Dense(1, activation='sigmoid'))
    logging.info("Model architecture finalized.")
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    logging.info("Training the model...")
    set_seed(conf['general']['seed_value'])
    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(learning_rate=conf['model']['learning_rate']), 
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=conf['training']['early_stop_patience'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                  factor=conf['training']['reduce_lr_factor'], 
                                  patience=conf['training']['reduce_lr_patience'], 
                                  min_lr=conf['training']['min_lr'])
    history = model.fit(X_train,
                        y_train,
                        batch_size=conf['training']['batch_size'],
                        epochs=conf['training']['epochs'],
                        validation_data=(X_val,y_val),
                        callbacks=[early_stopping, reduce_lr])
    mlflow.log_param("model_name", conf['exp_1']['model_name'])
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
    for metric_name, value in metrics.items():
        mlflow.log_metric(metric_name, value)

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    os.umask(0o002)
    set_seed(conf['general']['seed_value'])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_name = conf['exp_1']['model_name']+'_'+datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    with mlflow.start_run(run_name=run_name):
        X_train, X_val, y_train, y_val = load_data()
        model = define_model()
        start_time = time.time()
        model = train_model(model, X_train, y_train, X_val, y_val)
        evaluate_model(model, X_val, y_val, time.time()-start_time)
        mlflow.keras.log_model(
            model=model,
            artifact_path=conf['exp_1']['artifact_path'],
            signature=infer_signature(X_val, model.predict(X_val)),
            registered_model_name=conf['exp_1']['model_name'],
        )
        mlflow.log_param("operating_system", platform.platform())
        mlflow.log_param("train_script_hash", file_hash(TRAIN_FILE_PATH))
        mlflow.log_artifact(os.path.join(ROOT_DIR, 'requirements.txt'), 
                            conf['exp_1']['artifact_path'])
        mlflow.log_artifact(os.path.join(ROOT_DIR, 'settings.json'), 
                            conf['exp_1']['artifact_path'])
        mlflow.log_artifact(TRAIN_FILE_PATH, conf['exp_1']['artifact_path']+\
                            conf['general']['code_dir'])

# Executing the script
if __name__ == "__main__":
    main()