from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from scripts.main_intial import main as run_initial_job

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 12, 1),
    "retries": 0,
}

with DAG(
    dag_id="hotel_initial_canonicalization",
    default_args=default_args,
    schedule_interval="@once",
    catchup=False,
    tags=["hotel_matching", "canonicalization", "initial"],
) as dag:

    initial_cluster_task = PythonOperator(
        task_id="initial_cluster_and_upsert",
        python_callable=run_initial_job,
    )

    initial_cluster_task
