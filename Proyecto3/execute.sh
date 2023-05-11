#!/bin/bash

echo "delete everything in microk8s ..."
microk8s kubectl delete --all daemonsets,replicasets,services,deployments,pods,rc,ingress --namespace=default
# restart systemctl
# sudo systemctl daemon-reload
sudo sh -c 'echo "estudiante ALL=(ALL) NOPASSWD: /bin/systemctl daemon-reload" > /etc/sudoers.d/mlflow'
sudo systemctl daemon-reload
chmod +x  /home/estudiante/repo/MLOPS/Proyecto3/
sudo systemctl enable /home/estudiante/repo/MLOPS/Proyecto3/mlflow.service
#sudo sh -c 'echo "estudiante ALL=(ALL) NOPASSWD: /bin/systemctl enable /home/estudiante/repo/MLOPS/Proyecto3/mlflow.service"'
echo "enable service mlflow..."

# Build the Docker Compose project
echo "Building the Docker Compose project..."
# kill kubectl services
# pkill -f kubectl

pkill -f api-train
pkill -f api-inference
pkill -f frontend
pkill -f adminer
pkill -f api-store-info

ps aux | grep api-train
ps aux | grep api-inference
ps aux | grep frontend
ps aux | grep adminer
ps aux | grep api-store-info

docker rmi leodocker2021/my-repo-mlops-api-inference:latest leodocker2021/my-repo-mlops-api-train:latest leodocker2021/my-repo-mlops-frontend:latest leodocker2021/my-repo-mlops-api-store-info:latest


# Define an array of service names
services=( "frontend" "api-train" "api-inference" "api-store-info")

# Check if at least one argument is passed
if [ "$#" -eq 0 ]; then
  echo "No service name provided. Deploying all services..."
else
  # Only deploy the specified service(s)
  services=( "$@" )
fi

# Define an array of corresponding Docker image names
images=( "frontend" "api-train" "api-inference" "api-store-info")

# Loop through the service names and corresponding image names
for (( i=0; i<${#services[@]}; i++ )); do
  service="${services[$i]}"
  image="${images[$i]}"
  echo "$image"
  echo "Building Docker image for $service..."
  docker-compose build "$service"

  echo "Tagging the built image with the latest tag..."
  docker tag "leodocker2021/my-repo-mlops-$service:latest" "leodocker2021/my-repo-mlops-$service:latest"

  echo "Pushing the tagged image to Docker Hub..."
  docker push "leodocker2021/my-repo-mlops-$service:latest"

  echo "Checking if MicroK8s cluster is running..."
  sleep 20s

done

if microk8s status | grep -q "microk8s is running"; then
    echo "MicroK8s is running"
    echo "Checking if MicroK8s kubectl get nodes are ready.."
    # Check if Kubernetes API server is ready
    sleep 10s
    # execute compose to generate yaml
    chmod u+w komposefiles/
    kompose convert -f docker-compose.yml -o komposefiles/ --volumes hostPath
    # Apply the Kubernetes manifests to MicroK8s
    echo "Applying the Kubernetes manifests to MicroK8s..."
    microk8s kubectl apply -f komposefiles/
    echo "Applying the Kubernetes manifest to MicroK8s..."
    #microk8s kubectl apply -f komposefiles/"$service".yaml
    sleep 15s
    while true; do
      # Get the status of all pods
      status=$(microk8s kubectl get pods)

      # Count the number of pods in the "Running" state
      running=$(echo "$status" | grep -c "Running")

      # Check if all 4 pods are in the "Running" state
      if [ "$running" -gt 5 ]; then
          echo "All pods are running"
          break
      else
          echo "Waiting for all pods to start"
          sleep 15
      fi
    done
    echo "executing k8_start.."
    source k8_start.sh
    sudo systemctl start mlflow.service
    echo "mlflow service started."
    exit 1
  else
      echo "MicroK8s is not running"
      exit 1
fi


# docker-compose build
docker image inspect leodocker2021/my-repo-mlops-api-inference:latest | grep -E 'Id|Created'
docker image inspect leodocker2021/my-repo-mlops-api-train:latest | grep -E 'Id|Created'
docker image inspect leodocker2021/my-repo-mlops-frontend:latest | grep -E 'Id|Created'
docker image inspect leodocker2021/my-repo-mlops-api-store-info:latest | grep -E 'Id|Created'








