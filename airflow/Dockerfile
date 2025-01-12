FROM apache/airflow:2.10.4

USER root

RUN apt-get update \
    && apt-get install --no-install-recommends -y openjdk-17-jre vim\
    && apt-get clean;

ENV JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64/"
RUN export JAVA_HOME

USER airflow

RUN pip install apache-airflow==${AIRFLOW_VERSION} \
    apache-airflow-providers-apache-spark==5.0.0 \
    scikit-learn==1.6.1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

USER root
