FROM apache/airflow:2.9.0-python3.10

USER root

# Install necessary system dependencies
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    wget \
    vim \
    gcc \
    g++

# Set the working directory
WORKDIR /opt/airflow

# Copy the requirements file and install Python dependencies
COPY requirements.txt .

# Change to the airflow user before copying code
USER airflow

RUN pip install --no-cache-dir -r requirements.txt


# Copy the DAG files and other necessary files into the container
COPY ./dags /opt/airflow/dags
COPY ./data /opt/airflow/data
COPY ./config /opt/airflow/config

# Set the AIRFLOW_HOME environment variable
ENV AIRFLOW_HOME=/opt/airflow
