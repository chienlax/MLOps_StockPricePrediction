# Dockerfile
ARG AIRFLOW_VERSION=2.8.1
ARG PYTHON_VERSION=3.11  
FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}

# Set user to root temporarily to install packages
USER root

# Install system dependencies if needed
# e.g., RUN apt-get update && apt-get install -y --no-install-recommends build-essential && apt-get clean

# Switch back to the airflow user
USER airflow

# Copy requirements first to leverage Docker cache
COPY requirements.txt /requirements.txt

# Install Python dependencies from requirements.txt
# Using --user might be necessary if running pip as the 'airflow' user encounters permission issues
# writing to site-packages, although it's generally preferred to install system-wide
# if the base image allows it or use a virtualenv inside the container (more complex).
# Let's try without --user first.
RUN pip cache purge && \
    pip install --no-cache-dir -r /requirements.txt

# Optional: Copy source code if needed inside image (we use mount for dev)
# COPY src/ /opt/airflow/src/