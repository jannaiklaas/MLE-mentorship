import mlflow
import sys
import os
from tensorflow.keras.models import load_model

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

output_path = os.path.join(ROOT_DIR, 'src', 'test_model.keras')

# Ensure the tracking URI is correctly set
  # Change to your actual server URI

# Define the model URI
model_uri = "models:/model_exp_3.keras@champion"  # Adjust based on actual registered model name and stage

def load_model(output_path):
    global model
    print(f"Attempting to load model with URI: {model_uri}")
    model = mlflow.keras.load_model(model_uri)
    model.save(output_path)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://0.0.0.0:5001")
    try:
        load_model(output_path)
    except Exception as e:
        print(f"Failed to load model: {str(e)}", file=sys.stderr)