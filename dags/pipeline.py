from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import json

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def print_training_summary(**context):
    """Print summary of training results"""
    print("=" * 80)
    print("TRAINING PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Execution Date: {context['ds']}")
    print(f"DAG Run ID: {context['run_id']}")
    print("\n✓ Data Preprocessing: COMPLETED")
    print("✓ PyTorch Lightning Multi-Node DDP Training: COMPLETED")
    print("\nNext Run: Check Airflow schedule")
    print("=" * 80)

with DAG(
    'distributed_data_pipeline',
    default_args=default_args,
    description='Complete pipeline: Spark preprocessing → PyTorch Lightning DDP training',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['spark', 'preprocessing', 'pytorch-lightning', 'ddp', 'distributed'],
) as dag:

    start_task = BashOperator(
        task_id='start_pipeline',
        bash_command='''
        echo "════════════════════════════════════════════════════════════════"
        echo "🚀 DISTRIBUTED DATA PIPELINE STARTED"
        echo "════════════════════════════════════════════════════════════════"
        echo "Date: $(date)"
        echo "Pipeline: Spark Preprocessing → PyTorch Lightning Multi-Node DDP"
        echo "════════════════════════════════════════════════════════════════"
        '''
    )

    check_spark_cluster = BashOperator(
        task_id='check_spark_cluster',
        bash_command="""
        echo "Checking Spark cluster health..."
        
        if docker exec spark-master curl -sf http://localhost:8080 > /dev/null; then
            echo "✓ Spark Master is running!"
            docker exec spark-master curl -s http://localhost:8080 | grep -o "Workers ([0-9]*)" || echo "Workers: 2"
        else
            echo "✗ ERROR: Spark Master is not accessible"
            exit 1
        fi
        """
    )

    spark_preprocessing = BashOperator(
        task_id='spark_preprocessing',
        bash_command="""
        echo "════════════════════════════════════════════════════════════════"
        echo "📊 STEP 1: SPARK DATA PREPROCESSING"
        echo "════════════════════════════════════════════════════════════════"
        
        docker exec spark-master /opt/spark/bin/spark-submit \
            --master spark://spark-master:7077 \
            --deploy-mode client \
            --conf spark.executor.memory=1g \
            --conf spark.driver.memory=1g \
            --conf spark.sql.adaptive.enabled=true \
            /opt/spark/jobs/preprocess.py
        
        echo ""
        echo "✓ Spark preprocessing completed successfully!"
        """,
        execution_timeout=timedelta(minutes=30)
    )

    verify_preprocessing = BashOperator(
        task_id='verify_preprocessing_output',
        bash_command="""
        echo "🔍 Verifying preprocessing output..."
        
        if docker exec spark-master test -d /opt/spark/data/processed; then
            echo "✓ Processed data directory found!"
            echo ""
            echo "Data files:"
            docker exec spark-master ls -lh /opt/spark/data/processed/
            echo ""
            echo "Total size:"
            docker exec spark-master du -sh /opt/spark/data/processed/
        else
            echo "✗ ERROR: Processed data directory not found!"
            exit 1
        fi
        """
    )

    check_pytorch_cluster = BashOperator(
        task_id='check_pytorch_cluster',
        bash_command="""
        echo "🔍 Checking PyTorch DDP cluster..."
        
        # Check master node
        if docker exec pytorch-master python3 -c "import torch; import pytorch_lightning; print('PyTorch version:', torch.__version__); print('Lightning version:', pytorch_lightning.__version__)"; then
            echo "✓ PyTorch Master node is healthy!"
        else
            echo "✗ ERROR: PyTorch Master node is not accessible"
            exit 1
        fi
        
        # Check worker node
        if docker exec pytorch-worker python3 -c "import torch; import pytorch_lightning; print('PyTorch version:', torch.__version__); print('Lightning version:', pytorch_lightning.__version__)"; then
            echo "✓ PyTorch Worker node is healthy!"
        else
            echo "✗ ERROR: PyTorch Worker node is not accessible"
            exit 1
        fi
        
        echo ""
        echo "✓ All PyTorch DDP nodes are ready!"
        """
    )

    copy_data_to_pytorch = BashOperator(
        task_id='copy_data_to_pytorch_nodes',
        bash_command="""
        echo "📦 Ensuring data is accessible to PyTorch nodes..."
        
        # The data is already shared via volumes, just verify
        if docker exec pytorch-master test -d /workspace/data/processed; then
            echo "✓ Data accessible on master node"
        else
            echo "✗ ERROR: Data not found on master node"
            exit 1
        fi
        
        if docker exec pytorch-worker test -d /workspace/data/processed; then
            echo "✓ Data accessible on worker node"
        else
            echo "✗ ERROR: Data not found on worker node"
            exit 1
        fi
        
        echo "✓ All nodes can access the data!"
        """
    )

    pytorch_ddp_training = BashOperator(
        task_id='pytorch_lightning_ddp_training',
        bash_command="""
        echo "════════════════════════════════════════════════════════════════"
        echo "🔥 STEP 2: PYTORCH LIGHTNING MULTI-NODE DDP TRAINING"
        echo "════════════════════════════════════════════════════════════════"
        
        # Run training on master node
        # The master node will coordinate with worker automatically via DDP
        docker exec pytorch-master python3 /workspace/jobs/train_lightning_ddp.py &
        MASTER_PID=$!
        
        # Give master a moment to initialize
        sleep 5
        
        # Run training on worker node
        docker exec pytorch-worker python3 /workspace/jobs/train_lightning_ddp.py &
        WORKER_PID=$!
        
        # Wait for both processes
        echo "Waiting for training to complete on both nodes..."
        wait $MASTER_PID
        MASTER_EXIT=$?
        wait $WORKER_PID
        WORKER_EXIT=$?
        
        if [ $MASTER_EXIT -eq 0 ] && [ $WORKER_EXIT -eq 0 ]; then
            echo ""
            echo "✓ PyTorch Lightning DDP training completed successfully on all nodes!"
            exit 0
        else
            echo ""
            echo "✗ ERROR: Training failed on one or more nodes"
            echo "Master exit code: $MASTER_EXIT"
            echo "Worker exit code: $WORKER_EXIT"
            exit 1
        fi
        """,
        execution_timeout=timedelta(hours=3)
    )

    verify_model = BashOperator(
        task_id='verify_model_checkpoint',
        bash_command="""
        echo "🔍 Verifying model checkpoint..."
        
        if docker exec pytorch-master test -d /workspace/data/models; then
            echo "✓ Models directory found!"
            echo ""
            echo "Model files:"
            docker exec pytorch-master ls -lh /workspace/data/models/
            echo ""
            
            if docker exec pytorch-master test -f /workspace/data/models/final_model.ckpt; then
                echo "✓ Final model checkpoint exists!"
            else
                echo "⚠ Warning: Final model checkpoint not found (checking for best-*.ckpt)..."
                # Check for any .ckpt file
                if docker exec pytorch-master ls /workspace/data/models/*.ckpt 1> /dev/null 2>&1; then
                    echo "✓ Found Lightning checkpoints."
                else 
                    echo "✗ ERROR: No checkpoints found!"
                    exit 1
                fi
            fi
        else
            echo "✗ ERROR: Models directory not found!"
            exit 1
        fi
        """
    )

    check_logs = BashOperator(
        task_id='check_tensorboard_logs',
        bash_command="""
        echo "📊 Checking TensorBoard logs..."
        
        if docker exec pytorch-master test -d /workspace/data/logs; then
            echo "✓ Logs directory found!"
        else
            echo "⚠ Logs directory not found (this is OK for first run)"
        fi
        """
    )

    generate_report = PythonOperator(
        task_id='generate_training_report',
        python_callable=print_training_summary,
        provide_context=True
    )

    cleanup_task = BashOperator(
        task_id='cleanup_and_prepare',
        bash_command="""
        echo "🧹 Cleanup..."
        # Keep only last 3 model checkpoints
        docker exec pytorch-master bash -c '
            cd /workspace/data/models 2>/dev/null || exit 0
            ls -t model-*.ckpt 2>/dev/null | tail -n +4 | xargs rm -f 2>/dev/null || true
        '
        echo "✓ Cleanup completed"
        """
    )

    end_task = BashOperator(
        task_id='end_pipeline',
        bash_command='''
        echo "════════════════════════════════════════════════════════════════"
        echo "✅ DISTRIBUTED DATA PIPELINE COMPLETED SUCCESSFULLY!"
        echo "════════════════════════════════════════════════════════════════"
        '''
    )

    # --- 3. THE TRIGGER TASK (Updated) ---
    trigger_deployment = TriggerDagRunOperator(
        task_id='trigger_azure_rollout',
        trigger_dag_id='azure_smart_rollout', 
        wait_for_completion=False 
    )

    # --- 4. DEPENDENCY CHAIN (Fixed) ---
    start_task >> check_spark_cluster >> spark_preprocessing >> verify_preprocessing >> \
    check_pytorch_cluster >> copy_data_to_pytorch >> pytorch_ddp_training >> \
    verify_model >> check_logs >> generate_report >> cleanup_task >> \
    end_task >> trigger_deployment