import os
import sys
import json
import time
import shap
import optuna
import hashlib
import platform
import requests
import tempfile
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, roc_auc_score, confusion_matrix
import mlflow
from mlflow.models import infer_signature

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

CONF_FILE = os.path.join(ROOT_DIR, 'settings.json')

# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

TRAIN_FILE_PATH =  os.path.join(ROOT_DIR, conf['exp_4']['script_file_path'])
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', conf['general']['fallback_uri'])
MLFLOW_EXPERIMENT_NAME = conf['general']['experiment_name']

def file_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    return file_hash


def get_latest_commit_hash(repo_owner, repo_name, file_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?path={file_name}&per_page=1"
    response = requests.get(url)
    commits = response.json()
    if commits:
        return commits[0]['sha']
    return None


def load_data():
    col_names = conf['data']['col_names']
    url = conf['data']['url']
    data = pd.read_csv(url, header=None, names=col_names)
    # Automatically obtain the latest commit hash from GitHub
    repo_owner = conf['data']['repo_owner']
    repo_name = conf['data']['repo_name']
    file_name = conf['data']['file_name']
    dataset = mlflow.data.from_pandas(data, source=url, name=file_name, targets=conf['data']['target'])
    commit_hash = get_latest_commit_hash(repo_owner, repo_name, file_name)
    if commit_hash:
        mlflow.log_param("dataset_commit_hash", commit_hash)
    mlflow.log_param("dataset_repo_owner", repo_owner)
    mlflow.log_param("dataset_repo_name", repo_name)
    mlflow.log_input(dataset, context="training")
    return data


def preprocess(data, col_with_zeroes=conf['data']['col_with_zeroes'], col_to_drop = None):
    target = conf['data']['target']
    class_distribution = data[target].value_counts(normalize=True).to_dict()
    data[col_with_zeroes] = data[col_with_zeroes].replace(0, np.NaN)
    if col_to_drop:
        data = data.drop(col_to_drop, axis=1)
    features_selected_for_modeling = data.drop("Outcome", axis=1).columns.tolist()

    train, val = train_test_split(data, 
                                  test_size=conf['general']['val_size'], 
                                  random_state=conf['general']['random_state'], 
                                  stratify=data[target])
    class_distribution_train = train[target].value_counts(normalize=True).to_dict()
    class_distribution_val = val[target].value_counts(normalize=True).to_dict()
   
    for dataset in [train, val]:
        dataset[dataset.columns.drop(target)] = \
            dataset.groupby(target)[dataset.columns.drop(target)].transform(lambda x: x.fillna(x.median()))
    X_train, y_train = train.drop(target, axis=1), train[target]
    X_val, y_val = val.drop(target, axis=1), val[target]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    mlflow.log_param("class_distribution_before_preprocessing", json.dumps(class_distribution))
    mlflow.log_param("columns_with_zero_replaced_with_NaN", col_with_zeroes)
    mlflow.log_param("features_selected_for_modeling", ', '.join(features_selected_for_modeling))
    mlflow.log_param("val_size", conf['general']['val_size'])
    mlflow.log_param("test_train_split_random_state", conf['general']['random_state'])
    mlflow.log_param("test_train_split_stratify", target)
    mlflow.log_param("train_data_with_labels_shape", str(train.shape))
    mlflow.log_param("val_data_with_labels_shape", str(val.shape))
    mlflow.log_param("class_distribution_after_preprocessing_train", json.dumps(class_distribution_train))
    mlflow.log_param("class_distribution_after_preprocessing_val", json.dumps(class_distribution_val))
    mlflow.log_param("imputation_strategy", conf['general']['imputation_strategy'])
    mlflow.log_param("scaler", conf['general']['scaler'])

    return X_train, y_train, X_val, y_val, features_selected_for_modeling


def objective(trial, X_train, y_train, X_val, y_val):
    with mlflow.start_run(nested=True): 
        params = {
            "max_iter": conf['general']['max_iter'],
            "random_state": conf['general']['random_state'],
            "C": trial.suggest_loguniform('C', 1e-3, 1e3),
            "loss": trial.suggest_categorical('loss', ['hinge', 'squared_hinge']),
            "class_weight": trial.suggest_categorical('class_weight', [None, 'balanced'])
        }
        model = LinearSVC(**params).fit(X_train, y_train)
    return f1_score(y_val, model.predict(X_val))


def plot_shap_values(model, X, feature_names):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    temp_dir = tempfile.mkdtemp()
    shap_plot_path = os.path.join(temp_dir, "shap_feature_importance.png")
    plt.savefig(shap_plot_path)
    plt.close()
    mlflow.log_artifact(shap_plot_path, conf['exp_4']['artifact_path']+\
                        conf['general']['figure_dir'])
    shutil.rmtree(temp_dir)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    temp_dir = tempfile.mkdtemp()
    confusion_matrix_path = os.path.join(temp_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    mlflow.log_artifact(confusion_matrix_path, conf['exp_4']['artifact_path']+\
                        conf['general']['figure_dir'])
    shutil.rmtree(temp_dir)


def evaluate_model(model, X_val, y_val, training_time):
    y_pred = model.predict(X_val)
    metrics = {
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'F1 Score': f1_score(y_val, y_pred),
        'AUC ROC': roc_auc_score(y_val, model.decision_function(X_val)), 
        'Training duration': training_time
    }
    for metric_name, value in metrics.items():
        mlflow.log_metric(metric_name, value)


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        data = load_data()
        X_train, y_train, X_val, y_val, feature_names = preprocess(data,
                                                                   col_to_drop=conf['exp_4']['col_to_drop'])
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=conf['general']['random_state']), 
                                    direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), 
                       n_trials=conf['general']['tuning_trials'])
        start_time = time.time()
        model = LinearSVC(**study.best_params).fit(X_train, y_train)
        
        evaluate_model(model, X_val, y_val, time.time()-start_time)

        plot_shap_values(model, X_train, feature_names)
        plot_confusion_matrix(y_val, model.predict(X_val))

        mlflow.log_param("optuna_RandomSampler_seed", conf['general']['random_state'])
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=conf['exp_4']['artifact_path'],
            signature=infer_signature(X_val, model.predict(X_val)),
            registered_model_name=conf['exp_4']['model_name'],
            input_example=X_val
        )
        mlflow.log_param("operating_system", platform.platform())
        mlflow.log_param("train_script_hash", file_hash(TRAIN_FILE_PATH))
        mlflow.set_tags(
        tags={
            "optimizer_engine": conf['general']['optimizer'],
            "model_architecture": conf['exp_4']['model_type'],
            "feature_set_version": conf['exp_4']['feature_set_version'],
        }
        )
        mlflow.log_params(study.best_params)

        mlflow.log_artifact(os.path.join(ROOT_DIR, 'requirements.txt'), 
                            conf['exp_4']['artifact_path'])
        mlflow.log_artifact(TRAIN_FILE_PATH, conf['exp_4']['artifact_path']+\
                            conf['general']['code_dir'])


if __name__ == "__main__":
    main()