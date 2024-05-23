import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

class Predictor:
    def __init__(self):
        self.model_path = '/opt/ml/model/trained_model.pth'
        self.model = self.load_model()

    def load_model(self):
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
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
        return predictions
