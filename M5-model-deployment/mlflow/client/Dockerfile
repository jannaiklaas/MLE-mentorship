FROM python:3.10

ARG EXPERIMENT_NUMBER

WORKDIR /app

COPY ./src/train/exp_${EXPERIMENT_NUMBER}/ ./src/train/exp_${EXPERIMENT_NUMBER}
COPY ./src/preprocess.py ./src
COPY ./src/utils.py ./src
COPY ./data/train/ ./data/train
COPY settings.json ./
COPY requirements.txt ./
COPY ./mlflow/client/entrypoint.sh /entrypoint.sh

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt \
    && chmod +x /entrypoint.sh

ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000

ENTRYPOINT ["/entrypoint.sh"]
CMD []