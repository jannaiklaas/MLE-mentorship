import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from mle_hw5_package.batch_inference.run_batch_inference import run_inference, save_preds

@pytest.fixture
def model_mock():
    mock = MagicMock()
    mock.predict.return_value = np.array([[0.9], [0.8]])
    return mock

@pytest.fixture
def mock_files(tmp_path):
    d = tmp_path / "inference_images"
    d.mkdir()
    (d / "image1.jpg").write_bytes(b"fake image data")
    (d / "image2.jpg").write_bytes(b"fake image data")
    return d

@patch('cv2.imread', return_value=np.ones((128, 128, 3), dtype=np.uint8))
@patch('mle_hw5_package.batch_inference.run_batch_inference.load_model')
@patch('mle_hw5_package.batch_inference.run_batch_inference.preprocess_images', return_value=np.zeros((2, 128, 128, 3)))
@patch('os.listdir')
@patch('os.path.isfile')
def test_run_inference(mock_isfile, mock_listdir, mock_preprocess_images, mock_load_model, mock_imread, model_mock, mock_files):
    mock_load_model.return_value = model_mock
    mock_listdir.return_value = ['image1.jpg', 'image2.jpg']
    mock_isfile.return_value = True

    # Ensure the inference directory is set to the mock_files path for this test
    
    run_inference(model_name='model_1.keras', inference_dir=str(mock_files))

# Ensure that the test data set up uses the mocked paths
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

    assert output_dir.exists()
    files = list(output_dir.iterdir())
    assert len(files) == 1
    assert files[0].suffix == '.csv'
