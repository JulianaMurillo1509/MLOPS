FROM apache/airflow:2.6.1

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        build-essential \
        gcc \
        g++ \
        git \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements_dag.txt .
RUN pip install --no-cache-dir -r requirements_dag.txt
