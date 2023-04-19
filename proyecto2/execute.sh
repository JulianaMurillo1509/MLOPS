#!/bin/bash
# Build the Docker Compose project
echo "Building the Docker Compose project..."
# kill kubectl services
# pkill -f kubectl

pkill -f api-train
pkill -f api-inference
pkill -f frontend
ps aux | grep api-train
ps aux | grep api-inference
ps aux | grep frontend


docker-compose build

# Tag the built images with the latest tag
echo "Tagging the built images with the latest tag..."
docker tag leodocker2021/my-repo-mlops-api-inference:latest   leodocker2021/my-repo-mlops-api-inference:latest
docker tag leodocker2021/my-repo-mlops-api-train:latest   leodocker2021/my-repo-mlops-api-train:latest
docker tag leodocker2021/my-repo-mlops-api-frontend:latest   leodocker2021/my-repo-mlops-api-frontend:latest

# Push the latest tagged image to Docker Hub
echo "Pushing the latest tagged image to Docker Hub..."
docker push leodocker2021/my-repo-mlops-api-inference:latest
docker push leodocker2021/my-repo-mlops-api-train:latest
docker push leodocker2021/my-repo-mlops-api-frontend:latest

echo "Checking if MicroK8s cluster is running..."
sleep 60s
if microk8s status | grep -q "microk8s is running"; then
    echo "MicroK8s is running"
    echo "Checking if MicroK8s kubectl get nodes are ready.."
    # Check if Kubernetes API server is ready
    sleep 90s
    kompose convert -f docker-compose.yml -o komposefiles/ --volumes hostPath
    if ! microk8s kubectl get pods | grep -q  "Running"; then
      echo "Kubernetes API server is not ready."
      exit 1
    fi
    sleep 60s
    # kompose convert -f docker-compose.yml -o komposefiles/ --volumes hostPath
    # Apply the Kubernetes manifests to MicroK8s
    echo "Applying the Kubernetes manifests to MicroK8s..."
    microk8s kubectl apply -f komposefiles/
    echo "executing k8_start.."
    source k8_start.sh
    exit 1
else
    echo "MicroK8s is not running"
    exit 1
fi




