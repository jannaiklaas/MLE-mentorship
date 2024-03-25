import os
import sys
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from keras.applications import VGG16
from keras import regularizers


# Define directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from src.preprocess import preprocess_images, augment_data
from src.utils import set_seed

TRAIN_DIR = os.path.join(ROOT_DIR, 'data', 'train')
TRAIN_IMAGES_PATH = os.path.join(TRAIN_DIR, 'train_images.npy')
TRAIN_LABELS_PATH = os.path.join(TRAIN_DIR, 'train_labels.npy')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
BASE_MODEL_URL = "https://raw.githubusercontent.com/zdata-inc/applied_deep_learning/master/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

def define_model():
    base_model = VGG16(weights=None, include_top=False, input_shape=(128, 128, 3))
    weights_path = tf.keras.utils.get_file(
        fname="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
        origin=BASE_MODEL_URL,
        cache_subdir='models',
        md5_hash=None
    )
    base_model.load_weights(weights_path)
    base_model.layers[0].trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    return base_model, predictions

def train_model(base_model, predictions, train_generator, val_generator):
    logging.info("Training the model...")
    set_seed(42)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-5), metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7)
    history = model.fit(
        train_generator,
        epochs=200,
        validation_data=val_generator,
        callbacks=[early_stopping, reduce_lr]
    )
    logging.info("Training complete.")
    return model

def evaluate_model(model, X, y_true, set_name, threshold=0.5, model_name="model_1"):
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
    path = os.path.join(RESULTS_DIR, 'metrics.txt')
    metrics_str = "\n".join([f"{key}: {value}" for key, value in metrics.items()])
    with open(path, "a") as file:
        file.write(metrics_str + "\n\n")
    logging.info(f"Metrics saved to {path}")

def save_model(model, path) -> None:
    """Saves the trained model to the specified path."""
    logging.info("Saving the model...")
    if not os.path.exists(path):
        os.makedirs(path)
    model_path = os.path.join(path, 'model_1.keras')
    model.save(model_path)  
    logging.info(f"Model saved to {model_path}")

def main():
    set_seed(42)
    """Main method."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    X = preprocess_images(TRAIN_IMAGES_PATH)
    y = np.load(TRAIN_LABELS_PATH, allow_pickle=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    logging.info(f"Model will be trained with {X_train.shape[0]} and validated with {X_val.shape[0]} images")
    train_generator, val_generator = augment_data(X_train, y_train, X_val, y_val)
    base_model, predictions = define_model()
    model = train_model(base_model, predictions, train_generator, val_generator)
    evaluate_model(model, X_val, y_val, set_name="Validation")
    save_model(model, MODEL_DIR)


# Executing the script
if __name__ == "__main__":
    main()
