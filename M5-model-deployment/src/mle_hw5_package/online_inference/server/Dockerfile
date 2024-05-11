# Server Dockerfile

FROM python:3.10

WORKDIR /app

# Ensure paths are correct relative to the context set in docker-compose.yml
COPY ./src/mle_hw5_package/online_inference/server/ /app/src/mle_hw5_package/online_inference/server/
COPY ./src/mle_hw5_package/models/ /app/src/mle_hw5_package/models/
COPY ./src/mle_hw5_package/preprocess.py /app/src/mle_hw5_package/
COPY ./src/mle_hw5_package/utils.py /app/src/mle_hw5_package/
COPY ./settings.json /app/
COPY ./requirements.txt /app/

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 1234

CMD ["python", "/app/src/mle_hw5_package/online_inference/server/run_online_inference.py"]
