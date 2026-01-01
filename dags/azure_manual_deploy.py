import os
import shutil
import logging
import mlflow
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint, Model, Environment, CodeConfiguration
from azure.identity import DefaultAzureCredential
import os

# --- CONFIG ---
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
AZURE_WORKSPACE = os.getenv("AZURE_WORKSPACE")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")
DEPLOY_DIR = os.getenv("DEPLOY_DIR")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")



def get_ml_client():
    credential = DefaultAzureCredential()
    return MLClient(credential, AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE)

# --- TASK 1 ---
def prepare_package(**context):
    if os.path.exists(DEPLOY_DIR): shutil.rmtree(DEPLOY_DIR)
    os.makedirs(DEPLOY_DIR)
    
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    client = mlflow.MlflowClient()
    # Find best run
    runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name("weather_forecasting").experiment_id],
        order_by=["metrics.val_loss ASC"], max_results=1
    )
    if not runs: raise ValueError("No models found!")
    
    # Download
    run_id = runs[0].info.run_id
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="best_checkpoints", dst_path=DEPLOY_DIR)
    
    # Move .ckpt and Create Scripts
    for root, _, files in os.walk(local_path):
        for file in files:
            if file.endswith(".ckpt"):
                shutil.copy(os.path.join(root, file), os.path.join(DEPLOY_DIR, "model.ckpt"))
                break
    
    # Generate score.py
    # Generate score.py (Fixed for Nested Path)
    with open(os.path.join(DEPLOY_DIR, "score.py"), "w") as f:
        f.write("""
import os
import json
import torch
import pytorch_lightning as pl
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherClassifier(pl.LightningModule):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)

model = None

def init():
    global model
    logger.info("Initializing model...")
    
    # Get Base Path
    base_path = os.getenv("AZUREML_MODEL_DIR")
    logger.info(f"Base path: {base_path}")
    
    # ---------------------------------------------------------
    # FIX: Handle nested 'deployment_staging' folder
    # ---------------------------------------------------------
    expected_path = os.path.join(base_path, "model.ckpt")
    nested_path = os.path.join(base_path, "deployment_staging", "model.ckpt")
    
    if os.path.exists(expected_path):
        model_path = expected_path
    elif os.path.exists(nested_path):
        model_path = nested_path
        logger.info(f"Found nested model at: {model_path}")
    else:
        # Fallback: Find it wherever it is
        logger.warning(f"Model not found at {expected_path} or {nested_path}. Searching...")
        for root, dirs, files in os.walk(base_path):
            if "model.ckpt" in files:
                model_path = os.path.join(root, "model.ckpt")
                logger.info(f"Found model at: {model_path}")
                break
    # ---------------------------------------------------------

    try:
        model = WeatherClassifier.load_from_checkpoint(model_path, input_dim=5)
        model.eval()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"CRITICAL ERROR LOADING MODEL: {str(e)}")
        raise e

def run(raw_data):
    try:
        data = json.loads(raw_data)
        with torch.no_grad():
            logits = model(torch.tensor(data["data"]))
            probs = torch.softmax(logits, dim=1).tolist()
        return {"probabilities": probs}
    except Exception as e:
        return {"error": str(e)}
""")
    # Generate conda.yaml
    with open(os.path.join(DEPLOY_DIR, "conda.yaml"), "w") as f:
        f.write("""
name: weather-env
channels: [conda-forge]
dependencies:
  - python=3.10
  - pip: [torch==2.1.0, pytorch-lightning==2.1.0, pandas, scikit-learn, azureml-inference-server-http]
""")

# --- TASK 2: FORCE DEPLOY ---
def force_deploy(**context):
    client = get_ml_client()
    
    # 1. Handle Endpoint State
    try:
        endpoint = client.online_endpoints.get(name=ENDPOINT_NAME)
        if endpoint.provisioning_state.lower() == "failed":
            logging.warning("Endpoint is in failed state. Deleting to recreate...")
            client.online_endpoints.begin_delete(name=ENDPOINT_NAME).wait()
            raise Exception("Recreate")
    except Exception:
        logging.info("Creating new endpoint...")
        endpoint = ManagedOnlineEndpoint(name=ENDPOINT_NAME, auth_mode="key")
        client.online_endpoints.begin_create_or_update(endpoint).result()

    # 2. Deploy Model (Blue)
    logging.info(f"Force deploying to '{DEPLOYMENT_NAME}'...")
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=Model(path=DEPLOY_DIR),
        code_configuration=CodeConfiguration(code=DEPLOY_DIR, scoring_script="score.py"),
        environment=Environment(conda_file=os.path.join(DEPLOY_DIR, "conda.yaml"), image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"),
        instance_type="Standard_DS2_v2", instance_count=1
    )
    client.online_deployments.begin_create_or_update(deployment).result()
    
    # 3. Set 100% Traffic Immediately
    logging.info("Setting traffic to 100%...")
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    client.online_endpoints.begin_create_or_update(endpoint).result()

# --- DAG ---
with DAG('azure_manual_deploy', start_date=datetime(2025,1,1), schedule_interval=None, tags=['azure', 'manual']) as dag:
    t1 = PythonOperator(task_id='prepare_package', python_callable=prepare_package)
    t2 = PythonOperator(task_id='force_deploy_100', python_callable=force_deploy, execution_timeout=timedelta(minutes=40))
    t1 >> t2