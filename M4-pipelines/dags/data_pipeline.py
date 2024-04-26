import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from datetime import datetime
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago

# Constants
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF_FILE = os.path.join(ROOT_DIR, 'config', 'settings.json')
with open(CONF_FILE, 'r') as file:
        conf = json.load(file)

DATA_DIR = os.path.join(ROOT_DIR, conf['directories']['data'])
RAW_DATA_DIR = os.path.join(DATA_DIR, conf['directories']['raw_data'])
PREPROCESSED_DATA_DIR = os.path.join(DATA_DIR, conf['directories']['preprocessed_data'])
INTERMEDIATE_DATA_DIR = os.path.join(PREPROCESSED_DATA_DIR, conf['directories']['intermediate_data'])
COMPLETE_DATA_DIR = os.path.join(PREPROCESSED_DATA_DIR, conf['directories']['complete_data'])

# Helper function to get month-year string
def get_month_year(date_block_num):
    sales_start_date = datetime(2015, 1, 1)
    target_date = sales_start_date + pd.DateOffset(months=date_block_num)
    return target_date.strftime("%m-%Y")

# Load configuration settings
def load_config(conf=conf):
    Variable.set("config", conf)

# Load specific CSV file
def load_csv(file_name, task_id, date_block_num=None):
    file_path = os.path.join(RAW_DATA_DIR, file_name)
    data_frame = pd.read_csv(file_path)
    
    if date_block_num is not None and 'date_block_num' in data_frame.columns:
        data_frame = data_frame[data_frame['date_block_num'] == date_block_num]

    month_year = get_month_year(date_block_num) if date_block_num is not None else ''
    output_dir = os.path.join(INTERMEDIATE_DATA_DIR, month_year, task_id)
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, file_name)
    data_frame.to_csv(output_file_path, index=False)
    return output_file_path

# Function to extract city information
def obtain_shop_city(output_dir, **kwargs):
    # Load the DataFrame from the previous task's output or source file
    shops = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='load_shops'))
    
    # Data processing logic
    shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', "shop_name"] = 'СергиевПосад ТЦ "7Я"'
    shops["city"] = shops.shop_name.str.split(" ").map(lambda x: x[0])
    shops.loc[shops.city == "!Якутск", "city"] = "Якутск"
    
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'shops_with_city.csv')
    shops.to_csv(output_file_path, index=False)
    return output_file_path

# Function to extract shop type information
def obtain_shop_type(output_dir, **kwargs):
    # Load the DataFrame from the previous task's output
    shops = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='obtain_shop_city'))
    
    # Data processing logic
    shops['shop_type'] = 'regular'
    shops.loc[shops['shop_name'].str.contains(r'ТЦ|ТК'), 'shop_type'] = 'tc'
    shops.loc[shops['shop_name'].str.contains(r'ТРЦ|ТРК|МТРЦ'), 'shop_type'] = 'mall'
    shops.loc[shops['shop_id'].isin([9, 20]), 'shop_type'] = 'special'
    shops.loc[shops['shop_id'].isin([12, 55]), 'shop_type'] = 'online'
    
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'shops_with_type.csv')
    shops.to_csv(output_file_path, index=False)
    return output_file_path

# Function to encode shop city and shop type
def encode_shop_city_and_type(output_dir, **kwargs):
    shops = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='obtain_shop_type'))
    shops["shop_type"] = LabelEncoder().fit_transform(shops["shop_type"])
    shops["shop_city"] = LabelEncoder().fit_transform(shops["city"])
    shops = shops[["shop_id", "shop_type", "shop_city"]]
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'shops_encoded.csv')
    shops.to_csv(output_file_path, index=False)
    return output_file_path

def obtain_category_group(output_dir, **kwargs):
    cats = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='load_item_categories'))
    cats['group'] = cats['item_category_name'].str.split('-').map(lambda x: x[0].strip().lower())
    cats.loc[cats.item_category_name.str.contains(r'Билеты'), "group"] = 'билеты'
    cats.loc[cats.item_category_name.str.contains(r'Чистые носители'), "group"] = 'чистые носители'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'item_categories_with_group.csv')
    cats.to_csv(output_file_path, index=False)
    return output_file_path

def obtain_category_subgroup(output_dir, **kwargs):
    cats = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='obtain_category_group'))
    cats['subgroup'] = cats['item_category_name'].str.split('-').map(lambda x: x[1].strip().lower() if len(x) > 1 else x[0].strip().lower())
    cats.loc[cats.item_category_name == 'Билеты (Цифра)', "subgroup"] = 'цифра'
    cats.loc[cats.item_category_name == 'Служебные - Билеты', "subgroup"] = 'служебные'
    cats.loc[cats.item_category_name.str.contains(r'шпиль'), "subgroup"] = 'шпиль'
    cats.loc[cats.item_category_name.str.contains(r'штучные'), "subgroup"] = 'шпиль'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'item_categories_with_subgroup.csv')
    cats.to_csv(output_file_path, index=False)
    return output_file_path

def encode_item_categories(output_dir, **kwargs):
    cats = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='obtain_category_subgroup'))
    cats['group'] = LabelEncoder().fit_transform(cats['group'])
    cats['subgroup'] = LabelEncoder().fit_transform(cats['subgroup'])
    cats = cats[['item_category_id','group', 'subgroup']]
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'item_categories_encoded.csv')
    cats.to_csv(output_file_path, index=False)
    return output_file_path

# Define DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}
dag = DAG(
    'data_preprocessing',
    default_args=default_args,
    description='A DAG to preprocess sales data',
    schedule_interval=timedelta(hours=1),
    start_date=days_ago(1),
    tags=['sales', 'preprocessing'],
)

# Define tasks
t1 = PythonOperator(
    task_id='load_config',
    python_callable=load_config,
    dag=dag,
)

t2 = PythonOperator(
    task_id='load_item_categories',
    python_callable=load_csv,
    op_kwargs={'file_name': 'item_categories.csv', 'task_id': 'load_item_categories', 'date_block_num': 0},
    dag=dag,
)

t3 = PythonOperator(
    task_id='load_items',
    python_callable=load_csv,
    op_kwargs={'file_name': 'items.csv', 'task_id': 'load_items', 'date_block_num': 0},
    dag=dag,
)

t4 = PythonOperator(
    task_id='load_shops',
    python_callable=load_csv,
    op_kwargs={'file_name': 'shops.csv', 'task_id': 'load_shops', 'date_block_num': 0},
    dag=dag,
)

t5 = PythonOperator(
    task_id='load_sales_train',
    python_callable=load_csv,
    op_kwargs={'file_name': 'sales_train.csv', 'task_id': 'load_sales_train', 'date_block_num': 0},  # Adjust dynamically as needed
    dag=dag,
)

t6 = PythonOperator(
    task_id='obtain_category_group',
    python_callable=obtain_category_group,
    provide_context=True,
    op_kwargs={
         'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, get_month_year(0), 'obtain_category_group')
         },
    dag=dag,
)

t7 = PythonOperator(
    task_id='obtain_category_subgroup',
    python_callable=obtain_category_subgroup,
    provide_context=True,
    op_kwargs={
         'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, get_month_year(0), 'obtain_category_subgroup')
         },
    dag=dag,
)

t8 = PythonOperator(
    task_id='encode_item_categories',
    python_callable=encode_item_categories,
    provide_context=True,
    op_kwargs={
         'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, get_month_year(0), 'encode_item_categories')
         },
    dag=dag,
)

t9 = PythonOperator(
    task_id='obtain_shop_city',
    python_callable=obtain_shop_city,
    provide_context=True,
    op_kwargs={
         'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, get_month_year(0), 'obtain_shop_city')
         },
    dag=dag,
)

t10 = PythonOperator(
    task_id='obtain_shop_type',
    python_callable=obtain_shop_type,
    provide_context=True,
    op_kwargs={
         'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, get_month_year(0), 'obtain_shop_type')
         },
    dag=dag,
)

t11 = PythonOperator(
    task_id='encode_shops',
    python_callable=encode_shop_city_and_type,
    provide_context=True,
    op_kwargs={
         'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, get_month_year(0), 'encode_shops')
         },
    dag=dag,
)

# Load configurations first
t1 >> [t2, t3, t4, t5]

t2 >> t6 >> t7 >> t8 #item cats

t4 >> t9 >> t10 >> t11 # shops
