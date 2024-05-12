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

COPY ./data/inference_images /opt/airflow/data/inference_images
COPY ./src/mle_hw5_package/batch_inference /opt/airflow/src/mle_hw5_package/batch_inference
COPY ./config/settings.json /opt/airflow/config
COPY ./requirements.txt /opt/airflow
COPY ./src/mle_hw5_package/models /opt/airflow/src/mle_hw5_package/models
COPY ./src/mle_hw5_package/preprocess.py /opt/airflow/src/mle_hw5_package

RUN pip install --no-cache-dir -r requirements.txt

ENV AIRFLOW_HOME=/opt/airflow
