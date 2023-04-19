#!/bin/bash

# Convert the Docker Compose file to Kubernetes manifests
echo "Converting the Docker Compose file to Kubernetes manifests..."
# pkill -f kubectl
pkill -f api-train
pkill -f api-inference
pkill -f frontend
ps aux | grep api-train
ps aux | grep api-inference
ps aux | grep frontend

kompose convert -f docker-compose.yml -o komposefiles/ --volumes hostPath

# Apply the Kubernetes manifests to MicroK8s
echo "Applying the Kubernetes manifests to MicroK8s..."
microk8s kubectl apply -f komposefiles/

echo "Forwarding traffic from the services to your local machine..."

echo "Checking if MicroK8s cluster is running..."

# Check if MicroK8s is running#
#if ! microk8s status --wait-ready --timeout 30s > /dev/null 2>&1; then
#  echo "MicroK8s is not running or failed to start within the timeout period of 60 seconds."
#  exit 1
#fi
sleep 30s

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
