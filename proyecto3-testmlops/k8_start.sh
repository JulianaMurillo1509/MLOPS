#!/bin/bash

# Convert the Docker Compose file to Kubernetes manifests
echo "Converting the Docker Compose file to Kubernetes manifests..."
# pkill -f kubectl
pkill -f api-train
pkill -f api-inference
pkill -f frontend
pkill -f api-store-info
pkill -f mlflow
pkill -f minio

ps aux | grep api-train
ps aux | grep api-inference
ps aux | grep frontend
ps aux | grep api-store-info

echo "Forwarding traffic from the services to your local machine..."
echo "Checking if MicroK8s kubectl get nodes are ready.."
# Check if Kubernetes API server is ready
if ! microk8s kubectl get nodes > /dev/null 2>&1; then
  echo "Kubernetes API server is not ready."
  exit 1
fi

sleep 30s

echo "MicroK8s cluster is running. Forwarding traffic from the services to your local machine..."
microk8s kubectl port-forward --address 0.0.0.0 service/api-train 8502:8502 &
microk8s kubectl port-forward --address 0.0.0.0 service/api-inference 8503:8503 &
microk8s kubectl port-forward --address 0.0.0.0 service/frontend 8501:8501 &
microk8s kubectl port-forward --address 0.0.0.0 service/adminer 8080:8080 &
microk8s kubectl port-forward --address 0.0.0.0 service/api-store-info 8504:8504 &
microk8s kubectl port-forward --address 0.0.0.0 service/mlflow 5000:5000 &
microk8s kubectl port-forward --address 0.0.0.0 service/minio 9000:9000 9001:9001 &
echo "everything ok..."
exit 1