import pytest
from flask_testing import TestCase
from unittest.mock import patch, MagicMock
import numpy as np
import json
import cv2
import sys
import os
from io import BytesIO

# Load the config settings required for testing
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

CONF_FILE = os.path.join(ROOT_DIR, 'config/settings.json')
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

from src.mle_hw5_package.online_inference.server.run_online_inference import app, get_model

class TestFlaskApp(TestCase):
    def create_app(self):
        app.config['TESTING'] = True
        return app

    def setUp(self):
        """Prepare anything before each test."""
        self.client = app.test_client()

    def tearDown(self):
        """Clean up after each test."""
        pass

    @patch('src.mle_hw5_package.online_inference.server.run_online_inference.model', create=True)
    def test_predict(self, mock_model):
        """Test /predict route."""
        # Set up the mock model with a return value for predict
        mock_model.predict.return_value = np.array([[0.1]])

        # Mock preprocessing to prevent filesystem operations
        with patch('src.mle_hw5_package.online_inference.server.run_online_inference.preprocess_images', return_value=np.zeros((1, 128, 128, 3))), \
             patch('numpy.load', return_value=np.zeros((128, 128, 3))), \
             patch('cv2.imdecode', return_value=np.ones((128, 128, 3), dtype=np.uint8)):
            # Simulate POST with an image file
            response = self.client.post(
                '/predict',
                content_type='multipart/form-data',
                data={'file': (BytesIO(b'image data'), 'test.jpg')}
            )

            # Print the response data to debug the error
            print('Response data:', response.data.decode())

            # Check the response
            self.assertEqual(response.status_code, 200)


    def test_predict_no_file(self):
        """Test /predict route with no file attached."""
        response = self.client.post('/predict')
        self.assertEqual(response.status_code, 400)
        self.assertIn('File not found in request', response.data.decode())

    def test_predict_empty_file(self):
        """Test /predict route with an empty file."""
        response = self.client.post(
            '/predict',
            content_type='multipart/form-data',
            data={'file': (BytesIO(b''), 'test.jpg')}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('File is empty', response.data.decode())
