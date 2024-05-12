# Client Dockerfile

FROM python:3.10

ARG IMAGE_PATH="inference_images/5.png"
ENV IMAGE_PATH=${IMAGE_PATH}

ENV DOCKER_ENV=TRUE

WORKDIR /app

# Ensure paths are correct relative to the context set in docker-compose.yml
COPY ./data/inference_images/ ./data/inference_images
COPY ./src/mle_hw5_package/online_inference/client/ ./src/mle_hw5_package/online_inference/client/
COPY ./src/mle_hw5_package/config ./src/mle_hw5_package/config
COPY ./src/mle_hw5_package/requirements.txt .src/mle_hw5_package/

RUN pip install requests

WORKDIR /app/src/mle_hw5_package/online_inference/client

CMD ["python", "example_request_online_inference.py"]
