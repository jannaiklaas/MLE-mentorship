import pytest
from unittest.mock import patch
import numpy as np
from io import BytesIO
import io
from flask_testing import TestCase
from PIL import Image


# Import the Flask app setup but delay importing anything that uses the model directly
from mle_hw5_package.online_inference.server.run_online_inference import app, get_model

def create_test_image():
    # Create an image using PIL
    image = Image.new('RGB', (128, 128), color = 'red')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)  # Move cursor back to the start of the file before reading it
    return img_byte_arr

class TestFlaskApp(TestCase):
    def create_app(self):
        app.config['TESTING'] = True
        return app

    @pytest.fixture(autouse=True)
    def mock_model(self):
        with patch('mle_hw5_package.online_inference.server.run_online_inference.model') as mock:
            mock.predict.return_value = np.array([[0.1]])
            yield mock

    def test_predict(self):
        """Test /predict route."""
        valid_image_bytes = create_test_image()  # Get bytes of a valid image
        response = self.client.post(
            '/predict',
            content_type='multipart/form-data',
            data={'file': (valid_image_bytes, 'test.png')}
        )
        if response.status_code != 200:
            print('Response data:', response.data.decode())  # This will print the error message from the server
        assert response.status_code == 200


    def test_predict_no_file(self):
        """Test /predict route with no file attached."""
        response = self.client.post('/predict')
        assert response.status_code == 400
        assert 'File not found in request' in response.data.decode()

    def test_predict_empty_file(self):
        """Test /predict route with an empty file."""
        response = self.client.post(
            '/predict',
            content_type='multipart/form-data',
            data={'file': (BytesIO(b''), 'test.png')}
        )
        assert response.status_code == 400
        assert 'File is empty' in response.data.decode()

