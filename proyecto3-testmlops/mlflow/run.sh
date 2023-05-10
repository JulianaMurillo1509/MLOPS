#!/bin/sh
# run.sh
mlflow server \
    --backend-store-uri  postgresql://myuser:mypassword@db:5432/mydatabase \
    --default-artifact-root s3://mlop-s3/artifacts \
    --host 0.0.0.0 \
    --serve-artifacts \
    --port 5000