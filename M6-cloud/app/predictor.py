import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from flask import Flask, request, jsonify

class Predictor:
    def __init__(self):
        self.model_path = '/opt/ml/model/trained_model.pth'
        self.backup_model_path = '/opt/program/models/trained_model.pth'
        self.model = self.load_model()

    def load_model(self):
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError("No model available.")
        model = torch.jit.load(self.model_path)
        model.eval()
        return model

    def is_model_loaded(self):
        return self.model is not None

    def preprocess(self, data: pd.DataFrame) -> DataLoader:
        """Convert a DataFrame to a DataLoader."""
        input_tensor = torch.tensor(data.values).float()
        dataset = TensorDataset(input_tensor)
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    def predict(self, data: pd.DataFrame):
        predictions = []
        infer_loader = self.preprocess(data)
        with torch.no_grad():
            for inputs in infer_loader:
                outputs = self.model(inputs[0])
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        # Convert numpy.int64 to native Python int
        return [int(pred) for pred in predictions]

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy."""
    health = Predictor().is_model_loaded()
    status = 200 if health else 404
    return jsonify(status=status)

@app.route('/invocations', methods=['POST'])
def predict():
    """Do an inference on a single batch of data."""
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    try:
        input_data = pd.read_csv(file)
    except Exception as e:
        return jsonify(error=str(e)), 400

    predictions = Predictor().predict(input_data)
    return jsonify(predictions=predictions)
