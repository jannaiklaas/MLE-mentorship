# Client Dockerfile

FROM python:3.10

ARG IMAGE_PATH
ENV DOCKER_ENV=TRUE

WORKDIR /app

# Ensure paths are correct relative to the context set in docker-compose.yml
COPY ./data/raw/ /app/data/raw
COPY ./src/mle_hw5_package/online_inference/client/ /app/src/mle_hw5_package/online_inference/client/
COPY ./settings.json /app/
COPY ./requirements.txt /app/

RUN pip install requests

CMD ["python", "/app/src/mle_hw5_package/online_inference/client/example_request_online_inference.py"]
