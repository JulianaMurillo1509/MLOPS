FROM apache/airflow:2.6.1
USER airflow
COPY requirements_dag.txt .
RUN pip install --no-cache-dir -r requirements_dag.txt
USER airflow