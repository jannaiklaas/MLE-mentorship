import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.offsets import Day, MonthBegin, MonthEnd

import re
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords

from datetime import datetime
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

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
    sales_start_date = datetime(2013, 1, 1)
    target_date = sales_start_date + pd.DateOffset(months=date_block_num)
    return target_date.strftime("%m-%Y")

# Load configuration settings
def load_config(conf=conf):
    Variable.set("config", conf)

# Load specific CSV file
def load_csv(file_name, date_block_num=None, output_dir=None):
    df = pd.read_csv(os.path.join(RAW_DATA_DIR, file_name))
    
    if date_block_num is not None and 'date_block_num' in df.columns:
        date_block_num = int(date_block_num)
        max_block = df['date_block_num'].max()
        if date_block_num > max_block:
            raise ValueError(f"No data available for date_block_num: {date_block_num}")
        df = df[df['date_block_num'] == date_block_num]

    month_year = get_month_year(date_block_num) if date_block_num is not None else ''
    if output_dir == None: 
        output_dir = os.path.join(INTERMEDIATE_DATA_DIR, month_year)
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, file_name)
    df.to_csv(output_file_path, index=False)
    return output_file_path

# Methods related to item_category.csv
def add_category_group(output_dir, **kwargs):
    cats = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='load_item_categories'))
    cats['group'] = cats['item_category_name'].str.split('-').map(lambda x: x[0].strip().lower())
    cats.loc[cats.item_category_name.str.contains(r'Билеты'), "group"] = 'билеты'
    cats.loc[cats.item_category_name.str.contains(r'Чистые носители'), "group"] = 'чистые носители'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'item_categories_with_group.csv')
    cats.to_csv(output_file_path, index=False)
    return output_file_path

def add_category_subgroup(output_dir, **kwargs):
    cats = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='add_category_group'))
    cats['subgroup'] = cats['item_category_name'].str.split('-').map(lambda x: x[1].strip().lower() if len(x) > 1 else x[0].strip().lower())
    cats.loc[cats.item_category_name == 'Билеты (Цифра)', "subgroup"] = 'цифра'
    cats.loc[cats.item_category_name == 'Служебные - Билеты', "subgroup"] = 'служебные'
    cats.loc[cats.item_category_name.str.contains(r'шпиль'), "subgroup"] = 'шпиль'
    cats.loc[cats.item_category_name.str.contains(r'штучные'), "subgroup"] = 'штучные'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'item_categories_with_subgroup.csv')
    cats.to_csv(output_file_path, index=False)
    return output_file_path

def encode_item_categories(output_dir, **kwargs):
    cats = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='add_category_subgroup'))
    cats['group'] = LabelEncoder().fit_transform(cats['group'])
    cats['subgroup'] = LabelEncoder().fit_transform(cats['subgroup'])
    cats = cats[['item_category_id','group', 'subgroup']]
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'item_categories_encoded.csv')
    cats.to_csv(output_file_path, index=False)
    return output_file_path

# Methods related to items.csv
def add_item_name_group(output_dir, sim_thresh=65, feature_name="item_name_group", **kwargs):
    def partialmatchgroups(items, sim_thresh=sim_thresh):
        def strip_brackets(string):
            string = re.sub(r"\(.*?\)", "", string)
            string = re.sub(r"\[.*?\]", "", string)
            return string
        
        items = items.copy()
        items["nc"] = items.item_name.apply(strip_brackets)
        items["ncnext"] = np.concatenate((items["nc"].to_numpy()[1:], np.array([""])))

        def partialcompare(s):
            return fuzz.partial_ratio(s["nc"], s["ncnext"])

        items["partialmatch"] = items.apply(partialcompare, axis=1)
        # Assign groups
        grp = 0
        for i in range(items.shape[0]):
            items.loc[i, "partialmatchgroup"] = grp
            if items.loc[i, "partialmatch"] < sim_thresh:
                grp += 1
        items = items.drop(columns=["nc", "ncnext", "partialmatch"])
        return items
    
    items = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='load_items'))
    items = partialmatchgroups(items)
    items = items.rename(columns={"partialmatchgroup": feature_name})
    items = items.drop(columns="partialmatchgroup", errors="ignore")

    items[feature_name] = items[feature_name].apply(str)
    items[feature_name] = items[feature_name].factorize()[0]
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'items_with_name_groups.csv')
    items.to_csv(output_file_path, index=False)
    return output_file_path

def add_first_word_features(output_dir, feature_name="artist_name_or_first_word", **kwargs):
    # This extracts artist names for music categories and adds them as a feature.
    def extract_artist(st):
        st = st.strip()
        if st.startswith("V/A"):
            artist = "V/A"
        elif st.startswith("СБ"):
            artist = "СБ"
        else:
            # Retrieves artist names using the double space or all uppercase pattern
            mus_artist_dubspace = re.compile(r".{2,}?(?=\s{2,})")
            match_dubspace = mus_artist_dubspace.match(st)
            mus_artist_capsonly = re.compile(r"^([^a-zа-я]+\s)+")
            match_capsonly = mus_artist_capsonly.match(st)
            candidates = [match_dubspace, match_capsonly]
            candidates = [m[0] for m in candidates if m is not None]
            # Sometimes one of the patterns catches some extra words so choose the shortest one
            if len(candidates):
                artist = min(candidates, key=len)
            else:
                # If neither of the previous patterns found something, use the dot-space pattern
                mus_artist_dotspace = re.compile(r".{2,}?(?=\.\s)")
                match = mus_artist_dotspace.match(st)
                if match:
                    artist = match[0]
                else:
                    artist = ""
        artist = artist.upper()
        artist = re.sub(r"[^A-ZА-Я ]||\bTHE\b", "", artist)
        artist = re.sub(r"\s{2,}", " ", artist)
        artist = artist.strip()
        return artist
    
    nltk.download('stopwords', quiet=True)
    items = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='add_item_name_group'))
    items = items.copy()
    all_stopwords = stopwords.words("russian")
    all_stopwords = all_stopwords + stopwords.words("english")

    def first_word(string):
        # This cleans the string of special characters, excess spaces and stopwords then extracts the first word
        string = re.sub(r"[^\w\s]", "", string)
        string = re.sub(r"\s{2,}", " ", string)
        tokens = string.lower().split()
        tokens = [t for t in tokens if t not in all_stopwords]
        token = tokens[0] if len(tokens) > 0 else ""
        return token
    
    music_categories = [55, 56, 57, 58, 59, 60]
    items.loc[items.item_category_id.isin(music_categories), feature_name] = items.loc[
        items.item_category_id.isin(music_categories), "item_name"
    ].apply(extract_artist)
    items.loc[items[feature_name] == "", feature_name] = "other music"
    items.loc[~items.item_category_id.isin(music_categories), feature_name] = items.loc[
        ~items.item_category_id.isin(music_categories), "item_name"
    ].apply(first_word)
    items.loc[items[feature_name] == "", feature_name] = "other non-music"
    items[feature_name] = items[feature_name].factorize()[0]
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'items_with_artists_first_word_of_item_name.csv')
    items.to_csv(output_file_path, index=False)
    return output_file_path

def add_item_name_length(output_dir, **kwargs):
    def clean_text(string):
        # Removes bracketed terms, special characters and extra whitespace
        string = re.sub(r"\[.*?\]", "", string)
        string = re.sub(r"\(.*?\)", "", string)
        string = re.sub(r"[^A-ZА-Яa-zа-я0-9 ]", "", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = string.lower()
        return string
    
    items = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='add_first_word_features'))
    items["item_name_cleaned_length"] = items["item_name"].apply(clean_text).apply(len)
    items["item_name_length"] = items["item_name"].apply(len)
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'items_with_name_length.csv')
    items.to_csv(output_file_path, index=False)
    return output_file_path

def merge_items_with_categories(output_dir, **kwargs):
    items = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='add_item_name_length'))
    cats = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='encode_item_categories'))
    merged_items = cats.merge(items[["item_id", 
                              "item_category_id", 
                              "item_name_group", 
                              "artist_name_or_first_word",
                              "item_name_cleaned_length",
                              "item_name_length"]], on="item_category_id", how="left")
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'merged_item_categories.csv')
    merged_items.to_csv(output_file_path, index=False)
    return output_file_path


# Methods related to shops.csv
def add_shop_city(output_dir, **kwargs):
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
def add_shop_type(output_dir, **kwargs):
    # Load the DataFrame from the previous task's output
    shops = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='add_shop_city'))
    
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
    shops = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='add_shop_type'))
    shops["shop_type"] = LabelEncoder().fit_transform(shops["shop_type"])
    shops["shop_city"] = LabelEncoder().fit_transform(shops["city"])
    shops = shops[["shop_id", "shop_type", "shop_city"]]
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'shops_encoded.csv')
    shops.to_csv(output_file_path, index=False)
    return output_file_path

# Methods for sales_train.csv
def remove_outliers(output_dir, **kwargs):
    sales = pd.read_csv(kwargs['ti'].xcom_pull(task_ids=f'process_date_block_{kwargs["num"]}.load_sales_train'))
    sales = sales[(sales.item_price < 50000 )& (sales.item_cnt_day < 1001)]
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'sales_without_outliers.csv')
    sales.to_csv(output_file_path, index=False)
    return output_file_path

def remove_neg_values(output_dir, **kwargs):
    sales = pd.read_csv(kwargs['ti'].xcom_pull(task_ids=f'process_date_block_{kwargs["num"]}.remove_outliers'))
    sales = sales[(sales.item_price > 0 )& (sales.item_cnt_day > 0)]
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'sales_without_neg_values.csv')
    sales.to_csv(output_file_path, index=False)
    return output_file_path

def correct_shop_id(output_dir, **kwargs):
    sales = pd.read_csv(kwargs['ti'].xcom_pull(task_ids=f'process_date_block_{kwargs["num"]}.remove_neg_values'))
    sales["shop_id"] = sales["shop_id"].replace({0: 57, 1: 58, 11: 10})
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'sales_correct_shops.csv')
    sales.to_csv(output_file_path, index=False)
    return output_file_path

def add_revenue(output_dir, **kwargs):
    sales = pd.read_csv(kwargs['ti'].xcom_pull(task_ids=f'process_date_block_{kwargs["num"]}.correct_shop_id'))
    sales['revenue'] = sales['item_cnt_day'] * sales['item_price']
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'sales_with_revenue.csv')
    sales.to_csv(output_file_path, index=False)
    return output_file_path

def merge_current_with_previous_sales(output_dir, **kwargs):
    current_sales = pd.read_csv(kwargs['ti'].xcom_pull(task_ids=f'process_date_block_{kwargs["num"]}.add_revenue'))

    # Determine the previous month's file path
    previous_month_year = get_month_year(kwargs["num"] - 1)
    previous_file_path = os.path.join(INTERMEDIATE_DATA_DIR, previous_month_year, 'sales_merged/combined_sales.csv')
    
    # previous_sales = pd.read_csv(kwargs['ti'].xcom_pull(task_ids=f'process_date_block_{kwargs["num"]-1}.merge_current_with_previous_sales'))
    
    if os.path.exists(previous_file_path):
        # Load previous data if it exists
        previous_sales = pd.read_csv(previous_file_path)
        # Concatenate current and previous month's data
        combined_sales = pd.concat([previous_sales, current_sales], ignore_index=True)
    else:
        combined_sales = current_sales

    # Save the combined data
    combined_file_path = os.path.join(output_dir, 'combined_sales.csv')
    os.makedirs(os.path.dirname(combined_file_path), exist_ok=True)
    combined_sales.to_csv(combined_file_path, index=False)

    return combined_file_path

def aggregate_sales_data(output_dir, **kwargs):
    # Load the sales data with revenue
    sales = pd.read_csv(kwargs['ti'].xcom_pull(task_ids=f'process_date_block_{kwargs["num"]}.merge_current_with_previous_sales'))
    agg_sales = sales.groupby(['date_block_num', 'shop_id', 'item_id']).agg(
        item_cnt_month=('item_cnt_day', 'sum'),
        revenue_month=('revenue', 'sum')
    ).reset_index()

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'aggregated_sales.csv')
    agg_sales.to_csv(output_file_path, index=False)
    return output_file_path

def merge_shops_with_sales(output_dir, **kwargs):
    sales = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='aggregate_sales_data'))
    shops = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='encode_shops'))
    merged = sales.merge(shops, on="shop_id", how="left")
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'merged_shops_with_sales.csv')
    merged.to_csv(output_file_path, index=False)
    return output_file_path

def merge_items_with_sales_and_shops(output_dir, **kwargs):
    sales = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='merge_shops_with_sales'))
    items = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='merge_items_with_categories'))
    merged = sales.merge(items, on="item_id", how="left")
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'merged_item_with_sales_and_shops.csv')
    merged.to_csv(output_file_path, index=False)
    return output_file_path

def add_shop_and_item_age(**kwargs):
    train = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='merge_items_with_sales_and_shops'))
    train["item_age"] = train.groupby("item_id")["date_block_num"].transform(lambda x: x - x.min())
    # Sales tend to plateau after 12 months
    train["new_item"] = train["item_age"] == 0
    train["new_item"] = train["new_item"].astype("int8")
    train["shop_age"] = (train.groupby("shop_id")["date_block_num"].transform(lambda x: x - x.min()).astype("int8"))
    output_dir = os.path.join(INTERMEDIATE_DATA_DIR, 'time_features')
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'added_shop_and_item_age.csv')
    train.to_csv(output_file_path, index=False)
    return output_file_path

def add_time_features(**kwargs):
    train = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='add_shop_and_item_age'))
    sales = pd.read_csv(kwargs['ti'].xcom_pull(task_ids=f'process_date_block_{kwargs["num"]}.merge_current_with_previous_sales'))
    sales["date"] = pd.to_datetime(sales["date"], format="%d.%m.%Y")
    sales = sales.merge(train[["item_id", "item_name_group"]], on="item_id", how="left")

    month_last_day = sales.groupby("date_block_num").date.max().rename("month_last_day")
    month_last_day[~month_last_day.dt.is_month_end] = (
        month_last_day[~month_last_day.dt.is_month_end] + MonthEnd()
    )
    month_first_day = sales.groupby("date_block_num").date.min().rename("month_first_day")
    month_first_day[~month_first_day.dt.is_month_start] = (
        month_first_day[~month_first_day.dt.is_month_start] - MonthBegin()
    )
    month_length = (month_last_day - month_first_day + Day()).rename("month_length")
    first_shop_date = sales.groupby("shop_id").date.min().rename("first_shop_date")
    first_item_date = sales.groupby("item_id").date.min().rename("first_item_date")
    first_shop_item_date = (
        sales.groupby(["shop_id", "item_id"]).date.min().rename("first_shop_item_date")
    )
    first_item_name_group_date = (
        sales.groupby("item_name_group").date.min().rename("first_name_group_date")
    )
    train = train.merge(month_first_day, left_on="date_block_num", right_index=True, how="left")
    train = train.merge(month_last_day, left_on="date_block_num", right_index=True, how="left")
    train = train.merge(month_length, left_on="date_block_num", right_index=True, how="left")
    train = train.merge(first_shop_date, left_on="shop_id", right_index=True, how="left")
    train = train.merge(first_item_date, left_on="item_id", right_index=True, how="left")
    train = train.merge(
        first_shop_item_date, left_on=["shop_id", "item_id"], right_index=True, how="left"
    )
    train = train.merge(
        first_item_name_group_date, left_on="item_name_group", right_index=True, how="left"
    )
    train["shop_open_days"] = train["month_last_day"] - train["first_shop_date"] + Day()
    train["item_first_sale_days"] = train["month_last_day"] - train["first_item_date"] + Day()
    train["item_in_shop_days"] = (
        train[["shop_open_days", "item_first_sale_days", "month_length"]].min(axis=1).dt.days
    )
    train = train.drop(columns="item_first_sale_days")
    train["item_cnt_day_avg"] = train["item_cnt_month"] / train["item_in_shop_days"]
    train["month_length"] = train["month_length"].dt.days
    train["shop_open_days"] = train["month_first_day"] - train["first_shop_date"]
    train["first_item_sale_days"] = train["month_first_day"] - train["first_item_date"]
    train["first_shop_item_sale_days"] = train["month_first_day"] - train["first_shop_item_date"]
    train["first_name_group_sale_days"] = train["month_first_day"] - train["first_name_group_date"]
    train["shop_open_days"] = train["shop_open_days"].dt.days.fillna(0).clip(lower=0)
    train["first_item_sale_days"] = (
        train["first_item_sale_days"].dt.days.fillna(0).clip(lower=0).replace(0, 9999)
    )
    train["first_shop_item_sale_days"] = (
        train["first_shop_item_sale_days"].dt.days.fillna(0).clip(lower=0).replace(0, 9999)
    )
    train["first_name_group_sale_days"] = (
        train["first_name_group_sale_days"].dt.days.fillna(0).clip(lower=0).replace(0, 9999)
    )
    def last_sale_days(matrix):
        last_shop_item_dates = []
        for dbn in range(1, kwargs["num"]+1):
            lsid_temp = (
                sales.query(f"date_block_num<{dbn}")
                .groupby(["shop_id", "item_id"])
                .date.max()
                .rename("last_shop_item_sale_date")
                .reset_index()
            )
            lsid_temp["date_block_num"] = dbn
            last_shop_item_dates.append(lsid_temp)

        last_shop_item_dates = pd.concat(last_shop_item_dates)
        matrix = matrix.merge(
            last_shop_item_dates, on=["date_block_num", "shop_id", "item_id"], how="left"
        )

        def days_since_last_feat(m, feat_name, date_feat_name, missingval):
            m[feat_name] = (m["month_first_day"] - m[date_feat_name]).dt.days
            m.loc[m[feat_name] > 2000, feat_name] = missingval
            m.loc[m[feat_name].isna(), feat_name] = missingval
            return m

        matrix = days_since_last_feat(
            matrix, "last_shop_item_sale_days", "last_shop_item_sale_date", 9999
        )

        matrix = matrix.drop(columns=["last_shop_item_sale_date"])
        return matrix
    train = last_sale_days(train)
    train = train.drop(
        columns=[
            "first_day",
            "month_first_day",
            "month_last_day",
            "first_shop_date",
            "first_item_date",
            "first_name_group_date",
            "item_in_shop_days",
            "first_shop_item_date",
            "month_length",
        ],
        errors="ignore",
    )
    output_dir = os.path.join(INTERMEDIATE_DATA_DIR, 'time_features')
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'add_time_features.csv')
    train.to_csv(output_file_path, index=False)
    return output_file_path

def add_price_features(**kwargs):
    # Get mean prices per month from train dataframe
    sales = pd.read_csv(kwargs['ti'].xcom_pull(task_ids=f'process_date_block_{kwargs["num"]}.merge_current_with_previous_sales'))
    train = pd.read_csv(kwargs['ti'].xcom_pull(task_ids='add_time_features'))
    price_features = sales.groupby(["date_block_num", "item_id"]).item_price.mean()
    price_features = pd.DataFrame(price_features)
    price_features = price_features.reset_index()
    # Calculate normalized differenced from mean category price per month
    price_features = price_features.merge(
        train[["item_id", "item_category_id"]], how="left", on="item_id"
    )
    price_features["norm_diff_cat_price"] = price_features.groupby(
        ["date_block_num", "item_category_id"]
    )["item_price"].transform(lambda x: (x - x.mean()) / x.mean())
    # Retain only the necessary features
    price_features = price_features[
        [
            "date_block_num",
            "item_id",
            "item_price",
            "norm_diff_cat_price",
        ]
    ]
    features = [
        "item_price",
        "norm_diff_cat_price",
    ]
    newnames = ["last_" + f for f in features]
    aggs = {f: "last" for f in features}
    renames = {f: "last_" + f for f in features}
    features = []
    for dbn in range(1, kwargs["num"]+1):
        f_temp = (
            price_features.query(f"date_block_num<{dbn}")
            .groupby("item_id")
            .agg(aggs)
            .rename(columns=renames)
        )
        f_temp["date_block_num"] = dbn
        features.append(f_temp)
    features = pd.concat(features).reset_index()
    train = train.merge(features, on=["date_block_num", "item_id"], how="left")
    output_dir = os.path.join(INTERMEDIATE_DATA_DIR, 'price_features')
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'added_price_features.csv')
    train.to_csv(output_file_path, index=False)
    return output_file_path

# Define DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}
dag = DAG(
    'data_preprocessing',
    default_args=default_args,
    description='A DAG to preprocess sales data',
    schedule_interval=timedelta(hours=1),
    start_date=days_ago(1),
    tags=['sales', 'preprocessing'],
)

start_month_block = 0
end_month_block = 3

# Define tasks
load_config_task = PythonOperator(
    task_id='load_config',
    python_callable=load_config,
    dag=dag,
)

load_item_categories_task = PythonOperator(
    task_id='load_item_categories',
    python_callable=load_csv,
    op_kwargs={'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, 'item_catgories'),
               'file_name': 'item_categories.csv'},
    dag=dag,
)

load_items_task = PythonOperator(
    task_id='load_items',
    python_callable=load_csv,
    op_kwargs={'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, 'items'),
               'file_name': 'items.csv'},
    dag=dag,
)

load_shops_task = PythonOperator(
    task_id='load_shops',
    python_callable=load_csv,
    op_kwargs={'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, 'shops'),
               'file_name': 'shops.csv'},
    dag=dag,
)

add_category_group_task = PythonOperator(
    task_id='add_category_group',
    python_callable=add_category_group,
    provide_context=True,
    op_kwargs={
         'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, 'item_catgories', 'add_category_group')
         },
    dag=dag,
)

add_category_subgroup_task = PythonOperator(
    task_id='add_category_subgroup',
    python_callable=add_category_subgroup,
    provide_context=True,
    op_kwargs={
         'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, 'item_catgories', 'add_category_subgroup')
         },
    dag=dag,
)

encode_item_categories_task = PythonOperator(
    task_id='encode_item_categories',
    python_callable=encode_item_categories,
    provide_context=True,
    op_kwargs={
         'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, 'item_catgories', 'encode_item_categories')
         },
    dag=dag,
)

add_item_name_group_task = PythonOperator(
    task_id='add_item_name_group',
    python_callable=add_item_name_group,
    provide_context=True,
    op_kwargs={
         'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, 'items', 'add_item_name_group')
         },
    dag=dag,
)

add_first_word_features_task = PythonOperator(
    task_id='add_first_word_features',
    python_callable=add_first_word_features,
    provide_context=True,
    op_kwargs={
         'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, 'items', 'add_first_word_features')
         },
    dag=dag,
)

add_item_name_length_task = PythonOperator(
    task_id='add_item_name_length',
    python_callable=add_item_name_length,
    provide_context=True,
    op_kwargs={
         'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, 'items', 'add_item_name_length')
         },
    dag=dag,
)

merge_items_with_categories_task = PythonOperator(
    task_id='merge_items_with_categories',
    python_callable=merge_items_with_categories,
    provide_context=True,
    op_kwargs={
         'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, 'merge_items_with_categories')
         },
    dag=dag,
)

add_shop_city_task = PythonOperator(
    task_id='add_shop_city',
    python_callable=add_shop_city,
    provide_context=True,
    op_kwargs={
         'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, 'shops', 'add_shop_city')
         },
    dag=dag,
)

add_shop_type_task = PythonOperator(
    task_id='add_shop_type',
    python_callable=add_shop_type,
    provide_context=True,
    op_kwargs={
         'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, 'shops', 'add_shop_type')
         },
    dag=dag,
)

encode_shops_task = PythonOperator(
    task_id='encode_shops',
    python_callable=encode_shop_city_and_type,
    provide_context=True,
    op_kwargs={
         'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, 'shops', 'encode_shops')
         },
    dag=dag,
)

aggregate_sales_data_task = PythonOperator(
    task_id='aggregate_sales_data',
    python_callable=aggregate_sales_data,
    provide_context=True,
    op_kwargs={
         'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, 'aggregate_sales_data'),
         'num': end_month_block
         },
    dag=dag,
)

for num in range(int(start_month_block), int(end_month_block) + 1):
    with TaskGroup(group_id=f'process_date_block_{num}', dag=dag) as tg:
        load_sales_train_task = PythonOperator(
        task_id='load_sales_train',
        python_callable=load_csv,
        op_kwargs={
            'file_name': 'sales_train.csv', 
            'date_block_num': num
            }, 
        dag=dag,
        )

        remove_outliers_task = PythonOperator(
            task_id='remove_outliers',
            python_callable=remove_outliers,
            provide_context=True,
            op_kwargs={
                'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, get_month_year(num), 'remove_outliers'),
                'num': num
                },
            dag=dag,
            )

        remove_neg_values_task = PythonOperator(
            task_id='remove_neg_values',
            python_callable=remove_neg_values,
            provide_context=True,
            op_kwargs={
                'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, get_month_year(num), 'remove_neg_values'),
                'num': num
                },
            dag=dag,
            )

        correct_shop_id_task = PythonOperator(
            task_id='correct_shop_id',
            python_callable=correct_shop_id,
            provide_context=True,
            op_kwargs={
                'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, get_month_year(num), 'correct_shop_id'),
                'num': num
                },
            dag=dag,
            )
        

        add_revenue_task = PythonOperator(
            task_id='add_revenue',
            python_callable=add_revenue,
            provide_context=True,
            op_kwargs={
                'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, get_month_year(num), 'add_revenue'),
                'num': num
                },
            dag=dag,
        )

        merge_current_with_previous_sales_task = PythonOperator(
            task_id='merge_current_with_previous_sales',
            python_callable=merge_current_with_previous_sales,
            provide_context=True,
            op_kwargs={
                'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, get_month_year(num), 'sales_merged'),
                'num': num
                },
            dag=dag,
        )

        load_sales_train_task >> remove_outliers_task >> remove_neg_values_task >> correct_shop_id_task
        
        correct_shop_id_task >> add_revenue_task >> merge_current_with_previous_sales_task

    
    load_config_task >> [load_item_categories_task, load_items_task, load_shops_task, load_sales_train_task]

    load_item_categories_task >> add_category_group_task >> add_category_subgroup_task >> encode_item_categories_task #item cats

    load_items_task >> add_item_name_group_task >> add_first_word_features_task >> add_item_name_length_task

    [encode_item_categories_task, add_item_name_length_task] >> merge_items_with_categories_task

    load_shops_task >> add_shop_city_task >> add_shop_type_task >> encode_shops_task >> tg

    tg >> aggregate_sales_data_task



merge_shops_with_sales_task = PythonOperator(
    task_id='merge_shops_with_sales',
    python_callable=merge_shops_with_sales,
    provide_context=True,
    op_kwargs={
        'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, 'merge_shops_with_sales')
        },
    dag=dag,
)

merge_items_with_sales_and_shops_task = PythonOperator(
    task_id='merge_items_with_sales_and_shops',
    python_callable=merge_items_with_sales_and_shops,
    provide_context=True,
    op_kwargs={
        'output_dir': os.path.join(INTERMEDIATE_DATA_DIR, 'merge_items_with_sales_and_shops')
    },
    dag=dag,
)

add_shop_and_item_age_task = PythonOperator(
    task_id='add_shop_and_item_age',
    python_callable=add_shop_and_item_age,
    provide_context=True,
    dag=dag,
) 

add_time_features_task = PythonOperator(
    task_id='add_time_features',
    python_callable=add_time_features,
    provide_context=True,
    op_kwargs={
        'num': end_month_block
    },
    dag=dag,
)

add_price_features_task = PythonOperator(
    task_id='add_price_features',
    python_callable=add_price_features,
    provide_context=True,
    op_kwargs={
        'num': end_month_block
    },
    dag=dag,
)

[encode_shops_task, aggregate_sales_data_task] >> merge_shops_with_sales_task
[merge_items_with_categories_task, merge_shops_with_sales_task] >> merge_items_with_sales_and_shops_task >> add_shop_and_item_age_task
add_shop_and_item_age_task >> add_time_features_task >> add_price_features_task
