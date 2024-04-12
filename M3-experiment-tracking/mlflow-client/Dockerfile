FROM python:3.10

ARG EXPERIMENT_NUMBER

WORKDIR /app

COPY ./experiment_${EXPERIMENT_NUMBER}/ ./experiment_${EXPERIMENT_NUMBER}
COPY settings.json ./
COPY requirements.txt ./
COPY mlflow-client/entrypoint.sh /entrypoint.sh

RUN pip install --no-cache-dir -r requirements.txt \
&& chmod +x /entrypoint.sh

ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000

ENTRYPOINT ["/entrypoint.sh"]
CMD []