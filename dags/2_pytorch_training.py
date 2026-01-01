from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'pytorch_training_pipeline',
    default_args=default_args,
    description='Step 2: PyTorch DDP Training',
    schedule_interval=None,  # Triggered externally by Spark DAG
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['pytorch', 'ddp', 'training'],
) as dag:

    start_training = BashOperator(
        task_id='start_training',
        bash_command='echo "🔥 TRAINING PIPELINE STARTED"'
    )

    # 1. Clean Zombie Processes (Crucial for DDP stability)
    clean_zombies = BashOperator(
        task_id='cleanup_zombies',
        bash_command="""
        echo "🧹 Cleaning up previous DDP processes..."
        # Using || true to prevent failure if no process is found
        docker exec pytorch-master pkill -9 -f train_lightning_ddp.py || true
        docker exec pytorch-worker pkill -9 -f train_lightning_ddp.py || true
        sleep 2
        """
    )

    check_cluster = BashOperator(
        task_id='check_gpu_cluster',
        bash_command="""
        docker exec pytorch-master python3 -c "import torch; print('Master Ready')" && \
        docker exec pytorch-worker python3 -c "import torch; print('Worker Ready')"
        """
    )

    # 2. Run Distributed Training
    pytorch_ddp_training = BashOperator(
        task_id='pytorch_ddp_training',
        bash_command="""
        echo "🚀 Starting Distributed Training..."
        
        # Start Master Node (Background)
        docker exec pytorch-master python3 /workspace/jobs/train_lightning_ddp.py &
        MASTER_PID=$!
        sleep 5
        
        # Start Worker Node (Background)
        docker exec pytorch-worker python3 /workspace/jobs/train_lightning_ddp.py &
        WORKER_PID=$!
        
        # Wait for completion
        wait $MASTER_PID
        M_EXIT=$?
        wait $WORKER_PID
        W_EXIT=$?
        
        if [ $M_EXIT -eq 0 ] && [ $W_EXIT -eq 0 ]; then
            echo "✓ Training Success"
            exit 0
        else
            echo "✗ Training Failed (Master: $M_EXIT, Worker: $W_EXIT)"
            exit 1
        fi
        """,
        execution_timeout=timedelta(hours=3)
    )

    # 3. Verify Model Exists
    verify_model = BashOperator(
        task_id='verify_model',
        bash_command="""
        if docker exec pytorch-master ls /workspace/data/models/*.ckpt 1> /dev/null 2>&1; then
            echo "✓ Checkpoint found"
        else
            echo "✗ No checkpoint created"
            exit 1
        fi
        """
    )

    # ➤ TRIGGER NEXT DAG: Deployment
    trigger_deployment = TriggerDagRunOperator(
        task_id='trigger_azure_rollout',
        trigger_dag_id='azure_automated_rollout',  # Triggers your existing deployment DAG
        wait_for_completion=False
    )

    start_training >> clean_zombies >> check_cluster >> pytorch_ddp_training >> verify_model >> trigger_deployment