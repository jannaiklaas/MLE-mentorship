import os
import json
import numpy as np
import sys
import pandas as pd
from keras.models import load_model
import cv2
from datetime import datetime
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy import DummyOperator

ROOT_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)))))
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PACKAGE_DIR)

CONF_FILE = os.path.join(PACKAGE_DIR, 'config/settings.json')
with open(CONF_FILE, 'r') as file:
        conf = json.load(file)
MODEL_DIR = os.path.join(ROOT_DIR, conf['directories']['models'])
DATA_DIR = os.path.join(ROOT_DIR, conf['directories']['data'])
INFERENCE_DIR = os.path.join(DATA_DIR, conf['directories']['inference_data'])

from preprocess import preprocess_images

def do_nothing(**kwargs):
    print("No data available for processing.")

def check_data_availability(inference_dir = INFERENCE_DIR):
    for filename in os.listdir(inference_dir):
        img_path = os.path.join(inference_dir, filename)
        if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return 'check_model_availability'      
    return 'end_task'

def check_model_availability(model_dir=MODEL_DIR, model_name=conf['model']['name']+conf['model']['extension']):
    model_path = os.path.join(model_dir, model_name) 
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}. "
                                "Please train the model first.")
        return 'end_task'
    return 'run_inference' 
    
def get_data(inference_dir = INFERENCE_DIR):
    images = []
    all_files_checked = True
    for filename in os.listdir(inference_dir):
        img_path = os.path.join(inference_dir, filename)
        if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Failed to load image: {filename}")
                all_files_checked = False
        else:
            print(f"Skipping non-image file: {filename}")
    return np.array(images)


def run_inference(model_dir=MODEL_DIR,
                  model_name=conf['model']['name']+conf['model']['extension'], 
                  inference_dir = INFERENCE_DIR,
                  output_path=ROOT_DIR):
    images_array = get_data()
    X = preprocess_images(images_array=images_array)
    model_path = os.path.join(model_dir, model_name) 
    model = load_model(model_path)
    preds=model.predict(X)
    output_dir = os.path.join(output_path, 'batch_predictions')
    save_preds(preds, inference_dir, output_dir)

def save_preds(preds, inference_dir, output_dir, **kwargs):
    results = []
    for idx, filename in enumerate(sorted(os.listdir(inference_dir))):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(inference_dir, filename)
            if os.path.isfile(file_path):
                # Create a dictionary for each file
                prob = preds[idx][0]
                diagnosis = 'yes' if prob > conf['training']['decision_threshold'] else 'no'
                results.append({
                    "file_name": filename,
                    "probability": prob,
                    "diagnosis": diagnosis
                })
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%d.%m.%Y-%H")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{timestamp}.csv")
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


# Define DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=conf['airflow']['retry_delay']),
}

dag = DAG(
    'brain_tumor_prediction',
    default_args=default_args,
    description='A DAG to predict brain tumor',
    schedule_interval=timedelta(hours=conf['airflow']['schedule_interval']),
    start_date=days_ago(conf['airflow']['start_days_ago']),
    tags=['brain_tumor', 'prediction', 'inference'],
)
check_data_task = BranchPythonOperator(
     task_id='check_data_availability',
     python_callable=check_data_availability,
     provide_context=True,
     dag=dag
)

check_model_task = BranchPythonOperator(
     task_id='check_model_availability',
     python_callable=check_model_availability,
     provide_context=True,
     dag=dag
)

run_inference_task = PythonOperator(
     task_id='run_inference',
     python_callable=run_inference,
     provide_context=True,
     dag=dag
)


end_task = DummyOperator(
    task_id='end_task',
    dag=dag
)

check_data_task >> [check_model_task, end_task]
check_model_task >> [run_inference_task, end_task]