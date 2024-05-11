import os
import json
import numpy as np
import sys
from keras.models import load_model

from datetime import datetime
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy import DummyOperator

ROOT_DIR = os.path.dirname(
os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__))))))
sys.path.append(ROOT_DIR)

CONF_FILE = os.path.join(ROOT_DIR, 'settings.json')
with open(CONF_FILE, 'r') as file:
        conf = json.load(file)
MODEL_DIR = os.path.join(ROOT_DIR, 'src/mle_hw5_package/models')
DATA_DIR = os.path.join(ROOT_DIR, conf['directories']['data'])
RAW_DATA_DIR = os.path.join(DATA_DIR, conf['directories']['raw_data'])

# Load 5-10 images at per run, make sure those images were not loaded previously  

def get_model(model_name='model_1.keras'):
    """Loads and returns a trained model."""
    global model
    model_path = os.path.join(MODEL_DIR, model_name)  # Adjusted for directory path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}. "
                                "Please train the model first.")
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Failed to load model")
        sys.exit(1)

# Load configuration settings
def load_config(conf=conf):
    Variable.set("config", conf)

# Dummy function to continue operation
def do_nothing(**kwargs):
    print("No more data available for processing.")

# Define DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
    'params': {
        'start_month_block': 0,
        'sales_start_date': '2013-01-01'
    }
}
dag = DAG(
    'data_preprocessing',
    default_args=default_args,
    description='A DAG to preprocess sales data',
    schedule_interval=timedelta(minutes=20),
    start_date=days_ago(1),
    tags=['sales', 'preprocessing'],
)

end_task = DummyOperator(
    task_id='end_task',
    dag=dag
)