import requests
from requests_aws4auth import AWS4Auth
from requests_toolbelt.multipart.encoder import MultipartEncoder
import boto3

# Initialize a boto3 session
session = boto3.Session()
credentials = session.get_credentials()
region = session.region_name or 'us-east-1'

# Get AWS credentials
auth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'sagemaker', session_token=credentials.token)

# Set the endpoint name and URL
endpoint_name = "hw6-endpoint"
endpoint_url = f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations"

# Prepare the file for upload
file_path = '/Users/iklaas/Desktop/MLE-mentorship/M6-cloud/data/iris_inference_data.csv'

# Use MultipartEncoder to construct the multipart/form-data request
m = MultipartEncoder(fields={'file': ('iris_inference_data.csv', open(file_path, 'rb'), 'text/csv')})

# Set headers
headers = {"Content-Type": m.content_type}

# Make the request
response = requests.post(endpoint_url, auth=auth, data=m, headers=headers)

# Print the response
print(response.text)