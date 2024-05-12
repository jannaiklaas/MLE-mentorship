FROM apache/airflow:2.9.0-python3.10

USER root

WORKDIR /opt/airflow

# Update and install necessary packages including pkg-config
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    wget \
    vim \
    gcc \
    g++ \
    libhdf5-dev \
    libgl1-mesa-glx \
    pkg-config \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY ./data/inference_images/ ./data/inference_images
COPY ./src/mle_hw5_package/batch_inference/ ./src/mle_hw5_package/batch_inference
COPY ./src/mle_hw5_package/config/settings.json ./src/mle_hw5_package/config
COPY ./src/mle_hw5_package/requirements.txt ./src/mle_hw5_package
COPY ./models ./models
COPY ./src/mle_hw5_package/preprocess.py ./src/mle_hw5_package

RUN pip install --no-cache-dir -r ./src/mle_hw5_package/requirements.txt

ENV AIRFLOW_HOME=/opt/airflow
