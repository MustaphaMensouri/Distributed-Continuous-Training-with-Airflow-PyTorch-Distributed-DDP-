from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'spark_etl_pipeline',
    default_args=default_args,
    description='Step 1: Spark Data Preprocessing',
    schedule_interval='@daily',  # Runs daily
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['spark', 'etl'],
) as dag:

    start_task = BashOperator(
        task_id='start_etl',
        bash_command='echo "🚀 SPARK ETL PIPELINE STARTED"'
    )

    check_spark_cluster = BashOperator(
        task_id='check_spark_cluster',
        bash_command="""
        if docker exec spark-master curl -sf http://localhost:8080 > /dev/null; then
            echo "✓ Spark Master is healthy"
        else
            echo "✗ Spark Master is down"
            exit 1
        fi
        """
    )

    spark_preprocessing = BashOperator(
        task_id='spark_preprocessing',
        bash_command="""
        echo "📊 Running Spark Job..."
        docker exec spark-master /opt/spark/bin/spark-submit \
            --master spark://spark-master:7077 \
            --deploy-mode client \
            --conf spark.executor.memory=1g \
            /opt/spark/jobs/preprocess.py
        """,
        execution_timeout=timedelta(minutes=30)
    )

    verify_preprocessing = BashOperator(
        task_id='verify_output',
        bash_command="""
        if docker exec spark-master test -d /opt/spark/data/processed; then
            echo "✓ Data processed successfully"
        else
            echo "✗ Data directory missing"
            exit 1
        fi
        """
    )

    # ➤ TRIGGER NEXT DAG: Training
    trigger_training = TriggerDagRunOperator(
        task_id='trigger_training_dag',
        trigger_dag_id='pytorch_training_pipeline',  # Must match the ID of the second DAG below
        wait_for_completion=False 
    )

    start_task >> check_spark_cluster >> spark_preprocessing >> verify_preprocessing >> trigger_training