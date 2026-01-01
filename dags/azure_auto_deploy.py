import os
import shutil
import logging
import mlflow
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, Model, Environment, CodeConfiguration
from azure.identity import DefaultAzureCredential
import os

# --- CONFIG ---
client_id = os.getenv("AZURE_SUBSCRIPTION_ID")
client_id = os.getenv("AZURE_RESOURCE_GROUP")
client_id = os.getenv("AZURE_WORKSPACE")
client_id = os.getenv("ENDPOINT_NAME")
client_id = os.getenv("DEPLOY_DIR")


def get_ml_client():
    return MLClient(DefaultAzureCredential(), AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE)

# --- TASK 1: PREPARE PACKAGE (Robust Version) ---
def prepare_package(**context):
    if os.path.exists(DEPLOY_DIR): shutil.rmtree(DEPLOY_DIR)
    os.makedirs(DEPLOY_DIR)
    
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    client = mlflow.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name("weather_forecasting").experiment_id],
        order_by=["metrics.val_loss ASC"], max_results=1
    )
    if not runs: raise ValueError("No models found!")
    
    run_id = runs[0].info.run_id
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="best_checkpoints", dst_path=DEPLOY_DIR)
    
    for root, _, files in os.walk(local_path):
        for file in files:
            if file.endswith(".ckpt"):
                shutil.copy(os.path.join(root, file), os.path.join(DEPLOY_DIR, "model.ckpt"))
                break
    
    # Generate Robust score.py
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
    base_path = os.getenv("AZUREML_MODEL_DIR")
    
    # Robust Path Logic
    expected_path = os.path.join(base_path, "model.ckpt")
    nested_path = os.path.join(base_path, "deployment_staging", "model.ckpt")
    
    if os.path.exists(expected_path):
        model_path = expected_path
    elif os.path.exists(nested_path):
        model_path = nested_path
    else:
        # Fallback search
        for root, dirs, files in os.walk(base_path):
            if "model.ckpt" in files:
                model_path = os.path.join(root, "model.ckpt")
                break
    
    model = WeatherClassifier.load_from_checkpoint(model_path, input_dim=5)
    model.eval()
    logger.info("Model loaded successfully!")

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
    
    with open(os.path.join(DEPLOY_DIR, "conda.yaml"), "w") as f:
        f.write("""
name: weather-env
channels: [conda-forge]
dependencies:
  - python=3.10
  - pip: [torch==2.1.0, pytorch-lightning==2.1.0, pandas, scikit-learn, azureml-inference-server-http]
""")

# --- TASK 2: SMART DEPLOY (Using DS2 to save Quota) ---
def deploy_new_slot(**context):
    client = get_ml_client()
    endpoint = client.online_endpoints.get(name=ENDPOINT_NAME)
    
    # Logic: If Blue exists, deploy Green. If nothing exists, deploy Blue.
    traffic = endpoint.traffic
    if not traffic:
        current_prod = "blue"
        new_slot = "blue"
    else:
        current_prod = max(traffic, key=traffic.get)
        new_slot = "green" if current_prod == "blue" else "blue"
    
    logging.info(f"Current Prod: {current_prod}. Deploying to: {new_slot}")
    
    # Deploy using Standard_DS2_v2 (2 vCPUs) to fit in quota
    deployment = ManagedOnlineDeployment(
        name=new_slot,
        endpoint_name=ENDPOINT_NAME,
        model=Model(path=DEPLOY_DIR),
        code_configuration=CodeConfiguration(code=DEPLOY_DIR, scoring_script="score.py"),
        environment=Environment(
            conda_file=os.path.join(DEPLOY_DIR, "conda.yaml"), 
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
        ),
        instance_type="Standard_DS2_v2",  # <--- CHANGED BACK TO DS2
        instance_count=1
    )
    client.online_deployments.begin_create_or_update(deployment).result()
    
    context['ti'].xcom_push(key='new_slot', value=new_slot)
    context['ti'].xcom_push(key='old_slot', value=current_prod)

# --- TASK 3, 4, 5: TRAFFIC SHIFTING ---
def start_shadow(**context):
    client = get_ml_client()
    new = context['ti'].xcom_pull(task_ids='deploy_new_slot', key='new_slot')
    old = context['ti'].xcom_pull(task_ids='deploy_new_slot', key='old_slot')
    if new == old: return

    endpoint = client.online_endpoints.get(name=ENDPOINT_NAME)
    endpoint.traffic = {old: 100, new: 0}
    endpoint.mirror_traffic = {new: 20} # Mirror 20%
    client.online_endpoints.begin_create_or_update(endpoint).result()

def start_canary(**context):
    client = get_ml_client()
    new = context['ti'].xcom_pull(task_ids='deploy_new_slot', key='new_slot')
    old = context['ti'].xcom_pull(task_ids='deploy_new_slot', key='old_slot')
    if new == old: return

    endpoint = client.online_endpoints.get(name=ENDPOINT_NAME)
    endpoint.mirror_traffic = {} 
    endpoint.traffic = {old: 90, new: 10} # Live 10%
    client.online_endpoints.begin_create_or_update(endpoint).result()

def full_rollout(**context):
    client = get_ml_client()
    new = context['ti'].xcom_pull(task_ids='deploy_new_slot', key='new_slot')
    old = context['ti'].xcom_pull(task_ids='deploy_new_slot', key='old_slot')
    
    endpoint = client.online_endpoints.get(name=ENDPOINT_NAME)
    endpoint.traffic = {new: 100}
    client.online_endpoints.begin_create_or_update(endpoint).result()
    
    if new != old:
        logging.info(f"Deleting old deployment: {old}")
        client.online_deployments.begin_delete(name=old, endpoint_name=ENDPOINT_NAME).wait()

# --- DAG DEFINITION ---
with DAG('azure_automated_rollout', start_date=datetime(2025,1,1), schedule_interval=None, tags=['azure', 'auto']) as dag:
    t1 = PythonOperator(task_id='prepare_package', python_callable=prepare_package)
    t2 = PythonOperator(task_id='deploy_new_slot', python_callable=deploy_new_slot, execution_timeout=timedelta(minutes=40))
    t3 = PythonOperator(task_id='shadow_traffic', python_callable=start_shadow)
    t4 = BashOperator(task_id='wait_shadow', bash_command="sleep 30")
    t5 = PythonOperator(task_id='canary_traffic', python_callable=start_canary)
    t6 = BashOperator(task_id='wait_canary', bash_command="sleep 30")
    t7 = PythonOperator(task_id='full_rollout', python_callable=full_rollout)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7