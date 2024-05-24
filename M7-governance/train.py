import os
import sys
import time
import json
import seaborn as sns
from scipy.sparse import load_npz
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, roc_auc_score, confusion_matrix

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'train_processed.npz')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'test_processed.npz')
METRICS_DIR = os.path.join(ROOT_DIR, 'metrics')

def load_preprocessed_data(train_path, test_path):
    train = load_npz(train_path)
    test = load_npz(test_path)
    return train, test

class Model():
    """
    Manages the training process including running training, evaluating 
    the model, and saving the trained model.
    """
    def __init__(self) -> None:
        self.model = LinearSVC(random_state=42, C=0.01, max_iter=5000)

    def run_training(self, train, test) -> None:
        """Runs the model training process including evaluation and saving."""
        X_train, y_train = train[:, :-1], train[:, -1].toarray().ravel()
        X_test, y_test = test[:, :-1], test[:, -1].toarray().ravel()
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.evaluate(self.model, X_test, y_test, time.time()-start_time)

    @staticmethod
    def evaluate(model: LinearSVC, X_test, y_test, duration) -> float:
        """Evaluates the trained model on a test set."""
        # Calculate metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, model.decision_function(X_test))
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC ROC': auc_roc,
            'Training duration (s)': duration
        }
        os.makedirs(METRICS_DIR, exist_ok=True)
        path = os.path.join(METRICS_DIR, 'metrics.json')
        with open(path, "a") as file:
            json.dump(metrics, file)
        Model.plot_confusion_matrix(y_test, y_pred)

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred) -> None:
        """Plots and saves confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.xaxis.set_ticklabels(['Negative', 'Positive'])
        ax.yaxis.set_ticklabels(['Negative', 'Positive']) 
        fig_path = os.path.join(METRICS_DIR, "confusion_matrix.png")
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)


def main():
    train, test = load_preprocessed_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
    model = Model()
    model.run_training(train, test)

# Executing the script
if __name__ == "__main__":
    main()