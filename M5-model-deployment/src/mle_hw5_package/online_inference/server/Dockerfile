# Server Dockerfile

FROM python:3.10

WORKDIR /app

# Ensure paths are correct relative to the context set in docker-compose.yml
COPY ./src/mle_hw5_package/online_inference/server ./src/mle_hw5_package/online_inference/server
COPY ./models ./models
COPY ./src/mle_hw5_package/preprocess.py ./src/mle_hw5_package
COPY ./src/mle_hw5_package/utils.py ./src/mle_hw5_package
COPY ./src/mle_hw5_package/config/settings.json ./src/mle_hw5_package/config
COPY ./src/mle_hw5_package/requirements.txt .src/mle_hw5_package/

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r .src/mle_hw5_package/requirements.txt

EXPOSE 1234

WORKDIR /app/src/mle_hw5_package/online_inference/server

CMD ["python", "run_online_inference.py"]
