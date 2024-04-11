FROM python:3.10

WORKDIR /app

COPY ./experiment_5/ ./experiment_5
COPY settings.json ./
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Define environment variable for MLflow tracking URI
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000

# Run train.py when the container launches
CMD ["python", "./experiment_5/train.py"]