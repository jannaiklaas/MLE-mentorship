import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json
import sys

# Load the config settings required for testing
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

CONF_FILE = os.path.join(ROOT_DIR, 'config/settings.json')
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

from src.mle_hw5_package.batch_inference.run_batch_inference import check_data_availability, run_inference, save_preds

@pytest.fixture
def mock_files(tmp_path):
    d = tmp_path / "inference_images"
    d.mkdir()
    (d / "image1.jpg").write_bytes(b"fake image data")
    (d / "image2.jpg").write_bytes(b"fake image data")
    return d

@patch('os.listdir', return_value=['image1.jpg', 'image2.jpg'])
@patch('os.path.isfile', return_value=True)
def test_check_data_availability(mock_isfile, mock_listdir, mock_files):
    assert check_data_availability(inference_dir=str(mock_files)) == 'check_model_availability'

@pytest.fixture
def model_mock():
    mock = MagicMock()
    mock.predict.return_value = np.array([[0.9], [0.8]])
    return mock

@patch('cv2.imread', return_value=np.ones((128, 128, 3), dtype=np.uint8))
@patch('src.mle_hw5_package.batch_inference.run_batch_inference.load_model')
@patch('src.mle_hw5_package.batch_inference.run_batch_inference.preprocess_images', return_value=np.zeros((2, 128, 128, 3)))
def test_run_inference(mock_preprocess_images, mock_load_model, mock_imread, model_mock, mock_files):
    mock_load_model.return_value = model_mock
    # Mock `os.listdir` and `os.path.isfile` to ensure only two files are read
    with patch('os.listdir', return_value=['image1.jpg', 'image2.jpg']), \
         patch('os.path.isfile', return_value=True):
        run_inference(model_name='model_1.keras', inference_dir=str(mock_files))


@pytest.fixture
def mock_preds():
    return np.array([[0.8], [0.9]])

def test_save_preds(tmp_path, mock_preds):
    inference_dir = tmp_path / "inference_images"
    inference_dir.mkdir()
    (inference_dir / "image1.jpg").write_bytes(b"fake image data")
    (inference_dir / "image2.jpg").write_bytes(b"fake image data")

    output_dir = tmp_path / "batch_predictions"
    save_preds(mock_preds, str(inference_dir), str(output_dir))

    print("Checking if output directory exists:", output_dir.exists())
    assert output_dir.exists()

    files = list(output_dir.iterdir())
    assert len(files) == 1
    assert files[0].suffix == '.csv'

    df = pd.read_csv(files[0])
    assert len(df) == 2

