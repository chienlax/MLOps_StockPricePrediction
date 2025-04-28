ARG AIRFLOW_VERSION=2.8.1
ARG PYTHON_VERSION=3.11

FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}

USER root

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean

# Switch back to the airflow user
USER airflow

# Copy requirements first to leverage Docker cache
COPY requirements.txt /requirements.txt

# Install Python dependencies without using --user flag
RUN pip install --no-cache-dir -r /requirements.txt