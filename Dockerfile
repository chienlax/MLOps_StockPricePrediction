ARG AIRFLOW_VERSION=2.8.1
ARG PYTHON_VERSION=3.11

FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean

USER airflow

COPY requirements.txt /requirements.txt

RUN pip cache purge && \
    pip install --no-cache-dir -r /requirements.txt