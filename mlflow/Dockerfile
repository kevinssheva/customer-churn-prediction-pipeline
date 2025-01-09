# https://github.com/sachua/mlflow-docker-compose

FROM python:3.11-slim-buster

RUN apt-get update && apt-get install -y \
    postgresql-client

# Install python packages
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

COPY ./init_mlflow_db.sh /init_mlflow_db.sh
RUN chmod +x /init_mlflow_db.sh
