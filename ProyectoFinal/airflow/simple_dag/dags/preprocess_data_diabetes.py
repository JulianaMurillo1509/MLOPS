from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from diabetes_functions.functions import clean_data, read_data,train_model

default_args = {
    'start_date': datetime(2023, 5, 25),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG('data_cleaning_dag', schedule_interval='0 23 * * *', default_args=default_args) as dag:
    read_data_task = PythonOperator(
        task_id='read_data',
        python_callable=read_data,
        op_kwargs={'data': 'Diabetes'}
    )

    clean_data_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
        provide_context=True,
        op_kwargs = {'data': 'Diabetes_clean'}
    )

    train_model_task = PythonOperator(
        task_id='train_models',
        python_callable=train_model,
        provide_context=True,
        op_kwargs={'data': 'Diabetes_clean'}
    )

    read_data_task >> clean_data_task
