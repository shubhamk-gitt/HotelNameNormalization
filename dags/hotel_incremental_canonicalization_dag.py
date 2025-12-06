from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from scripts.main_incremental import main as run_incremental_job

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 12, 1),
    "retries": 1,
}

with DAG(
    dag_id="hotel_incremental_canonicalization",
    default_args=default_args,
    schedule_interval="0 1 * * *", #run every night
    catchup=False,
    tags=["hotel_matching", "canonicalization", "incremental"],
) as dag:

    incremental_task = PythonOperator(
        task_id="incremental_match_and_upsert",
        python_callable=run_incremental_job,
    )

    incremental_task
