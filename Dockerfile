# Dockerfile
ARG AIRFLOW_VERSION=2.8.1
ARG PYTHON_VERSION=3.11  
FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}

# Set user to root temporarily to install packages
USER root

# Switch back to the airflow user
USER airflow

# Copy requirements first to leverage Docker cache
COPY requirements.txt /requirements.txt

RUN pip cache purge && \
    pip install --no-cache-dir -r /requirements.txt

# Optional: Copy source code if needed inside image (we use mount for dev)
# COPY src/ /opt/airflow/src/