import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.append("/opt/airflow")  # lets Python find src folder inside the container

from src.prepare import prepare_data
from src.preprocess import preprocess
from src.train import train
from src.evaluate import evaluate

with DAG(
    dag_id="wellsense_training",
    start_date=datetime(2026, 11, 1),
    schedule="@weekly",                
    catchup=False,                   
    default_args={"retries": 1},  # retry a failed task once
) as dag:

    t1 = PythonOperator(task_id="prepare_data", python_callable=prepare_data)
    t2 = PythonOperator(task_id="preprocess", python_callable=preprocess)
    t3 = PythonOperator(task_id="train", python_callable=train)
    t4 = PythonOperator(task_id="evaluate", python_callable=evaluate)

    t1 >> t2 >> t3 >> t4