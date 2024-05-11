import os
import sys
import json
import time
import mlflow
import logging
import hashlib
from datetime import datetime
import platform
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

TRAIN_FILE_PATH =  os.path.join(ROOT_DIR, conf['exp_3']['script_file_path'])
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
    mlflow.log_param("model_name", conf['exp_3']['model_name'])
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
    run_name = conf['exp_3']['model_name']+'_'+datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    with mlflow.start_run(run_name=run_name):
        X_train, X_val, y_train, y_val = load_data()
        train_generator, val_generator = augment_data(X_train, y_train, X_val, y_val)
        base_model, predictions = define_model()
        start_time = time.time()
        model = train_model(base_model, predictions, train_generator, val_generator)
        evaluate_model(model, X_val, y_val, time.time()-start_time)
        mlflow.keras.log_model(
            model=model,
            artifact_path=conf['exp_3']['artifact_path'],
            signature=infer_signature(X_val, model.predict(X_val)),
            registered_model_name=conf['exp_3']['model_name']
        )
        mlflow.log_param("operating_system", platform.platform())
        mlflow.log_param("train_script_hash", file_hash(TRAIN_FILE_PATH))
        mlflow.log_artifact(os.path.join(ROOT_DIR, 'requirements.txt'), 
                            conf['exp_3']['artifact_path'])
        mlflow.log_artifact(os.path.join(ROOT_DIR, 'settings.json'), 
                            conf['exp_3']['artifact_path'])
        mlflow.log_artifact(TRAIN_FILE_PATH, conf['exp_3']['artifact_path']+\
                            conf['general']['code_dir'])



# Executing the script
if __name__ == "__main__":
    main()
