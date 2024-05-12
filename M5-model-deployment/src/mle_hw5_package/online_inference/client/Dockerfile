# Client Dockerfile

FROM python:3.10

ARG IMAGE_PATH="inference_images/5.png"
ENV IMAGE_PATH=${IMAGE_PATH}

ENV DOCKER_ENV=TRUE

WORKDIR /app

# Ensure paths are correct relative to the context set in docker-compose.yml
COPY ./data/inference_images/ /app/data/inference_images
COPY ./src/mle_hw5_package/online_inference/client/ /app/src/mle_hw5_package/online_inference/client/
COPY ./config/settings.json /app/config/
COPY ./requirements.txt /app/

RUN pip install requests

CMD ["python", "/app/src/mle_hw5_package/online_inference/client/example_request_online_inference.py"]
