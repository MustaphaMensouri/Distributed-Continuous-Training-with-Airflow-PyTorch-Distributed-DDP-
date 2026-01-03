# Distributed Continuous Training with Airflow + PyTorch DDP
---

## Overview

This project implements a scalable end-to-end **MLOps pipeline** using a **microservices architecture**.  
Unlike monolithic scripts, the system isolates components to efficiently handle **big data processing**, **distributed training**, and **cloud deployment**.

The pipeline is designed to be **modular**, **scalable**, and **production-oriented**, following modern MLOps best practices.

### Key Features

- **Orchestration:** Apache Airflow manages and schedules all workflows  
- **Big Data Processing:** Apache Spark cluster for distributed ETL  
- **Distributed Training:** PyTorch Distributed Data Parallel (DDP)  
- **Deployment:** Automated deployment to Azure Machine Learning (Online Endpoints)  
- **Tracking:** MLflow for experiment tracking and model registry  

---

## Architecture

The system is composed of multiple Docker containers, each responsible for a specific task in the pipeline.

| Service | Description | Local URL |
|------|------------|-----------|
| Airflow | Workflow Orchestrator (Scheduler & Webserver) | http://localhost:8080 |
| Spark Master | Cluster Manager for Spark | http://localhost:8081 |
| Spark Worker | Distributed Processing Node | N/A |
| PyTorch Master | DDP Master Node (Rank 0) | N/A |
| PyTorch Worker | DDP Worker Node (Rank 1) | N/A |
| MLflow | Experiment Tracking & Model Registry | http://localhost:5000 |

---

## Prerequisites

Before running the project, ensure your machine meets the following requirements:

- **Operating System:** Linux, macOS, or Windows (WSL2)
- **Software Requirements:**
  - Docker Engine / Docker Desktop (v20.10 or later)
  - Docker Compose (v2.0 or later)
- **Hardware Requirements:**
  - Minimum 8 GB RAM (16 GB recommended)
  - At least 20 GB of free disk space
- **Cloud Requirement:**
  - An active Azure account (required for deployment phase)

---

## Installation & Setup

### 1 Clone the Repository

```bash
git clone https://github.com/MustaphaMensouri/Distributed-Continuous-Training-with-Airflow-PyTorch-Distributed-DDP-.git
cd Distributed-Continuous-Training-with-Airflow-PyTorch-Distributed-DDP-
```

### 2 Prepare Data Directories

The project uses Docker bind mounts to share data between containers. You must create the directory structure before starting Docker Compose.

```bash
mkdir -p data/raw data/processed data/registry
```

📌 **Important:** Place your dataset file `weather.csv` inside: `data/raw/`

### 3 Azure Configuration (.env)

Create a `.env` file in the root directory of the project. Docker Compose will automatically read this file to authenticate with Azure.

```bash
AIRFLOW_UID=1000
# --- Airflow Security ---
AIRFLOW_FERNET_KEY=46BKJoQYlPPOexq0OhDZnIlNepKFf87WFwLbfzqDDho=
AIRFLOW_SECRET_KEY=fix_my_logs_please_12345
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_PASSWORD=admin

# --- Database Credentials (Airflow) ---
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow

# --- Database Credentials (MLflow) ---
MLFLOW_DB_USER=mlflow
MLFLOW_DB_PASSWORD=mlflow
MLFLOW_DB_NAME=mlflow

# Azure Authentication
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_TENANT_ID=your-tenant-id
AZURE_SUBSCRIPTION_ID=your-subscription-id

# Project Configuration
AZURE_RESOURCE_GROUP=project_mlops
AZURE_WORKSPACE=azure_ml_project_mlops
ENDPOINT_NAME=weather-api
DEPLOY_DIR=/opt/airflow/data/deployment_staging
DEPLOYMENT_NAME=blue
```

⚠️ **Security Note:** Do NOT commit this file to Git.

### 4 Build and Launch the Infrastructure

Build images and start all services using Docker Compose:

```bash
docker-compose up --build -d
```

⏳ The first build may take several minutes.

---

## 🏃‍♂️ Running the Pipeline

### Phase 1: Data Processing & Distributed Training

1. **Open Airflow Web UI:** http://localhost:8080

   Login credentials:
   - Username: `admin`
   - Password: `admin`

2. **Enable and trigger the DAG:** `spark_preprocessing_pipeline`

   This DAG:
   - Cleans and preprocesses data using Apache Spark
   - Saves processed data to `data/processed`
   - Automatically triggers the training pipeline

3. **Distributed training starts via:** `distributed_training_pipeline`
   - PyTorch DDP launches on master and worker nodes simultaneously
   - Training runs in parallel across containers

4. **Monitor training metrics and experiments in MLflow:** http://localhost:5000

### Phase 2: Deployment Pipeline (Azure)

Once a model is successfully trained, use the Airflow deployment DAGs:

- **`azure_manual_deploy`**
  - Creates the Azure Online Endpoint
  - Performs the initial deployment

- **`azure_automated_rollout`**
  - Executes Blue/Green deployment
  - Supports Shadow and Canary traffic shifting

---

## ☁️ Azure Setup Details

### Create Required Azure Resources

- Resource Group: `project_mlops`
- Azure ML Workspace: `azure_ml_project_mlops`

### Generate Service Principal Credentials

Use Azure CLI to generate credentials for Airflow.

```bash
# Get Subscription ID
az account show --query id --output tsv

# Create Service Principal
az ad sp create-for-rbac \
  --name "airflow-sp" \
  --role contributor \
  --scopes /subscriptions/<YOUR_SUBSCRIPTION_ID>/resourceGroups/project_mlops \
  --sdk-auth
```

Copy the generated values into your `.env` file.

---

## 📂 Project Structure

```
├── docker-compose.yml        # Infrastructure definition
├── Dockerfile.pytorch        # Custom PyTorch DDP image
├── .env                      # Secrets (not committed)
├── dags/                     # Airflow DAGs
│   ├── preprocessing.py
│   ├── distributed_training.py
│   └── azure_auto_deploy.py
├── jobs/                     # Spark & PyTorch jobs
│   ├── preprocess.py
│   └── train_lightning_ddp.py
├── data/                     # Shared volume
│   ├── raw/                  # Input dataset (weather.csv)
│   ├── processed/            # Spark output
│   └── registry/             # Model checkpoints
└── logs/                     # System logs
```

---

## 👥 Authors

**Prepared by:**
- Mustapha Mensouri
- Nassim Ait Dihim
- Abderahman El-Hamidy

**Supervised by:**
- Pr. Fahd Kalloubi
