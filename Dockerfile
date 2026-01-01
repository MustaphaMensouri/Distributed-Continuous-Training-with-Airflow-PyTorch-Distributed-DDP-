# Use the same Airflow image as base
FROM apache/airflow:2.7.1-python3.10

# Switch to root to install system tools (needed for some Azure/pandas dependencies)
USER root
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && apt-get clean

# Switch back to airflow user to install Python packages
USER airflow

# Install the heavy libraries ONCE during build
RUN pip install --no-cache-dir \
    azure-ai-ml \
    azure-identity \
    mlflow \
    pandas \
    scikit-learn