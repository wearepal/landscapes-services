#!/bin/bash

# Define variables
NETWORK_NAME="landscapes_services"
CONTAINER_NAME="landscapes-services"
DOCKER_IMAGE_NAME="landscapes-services"
EXPOSED_PORT="5001"

# Check if the network exists, if not, create it
if ! docker network ls | grep -q "$NETWORK_NAME"; then
  echo "Network $NETWORK_NAME does not exist, creating it..."
  docker network create "$NETWORK_NAME"
else
  echo "Network $NETWORK_NAME already exists."
fi

# Stop and remove the existing container if running
if docker ps -a | grep -q "$CONTAINER_NAME"; then
  echo "Stopping and removing the old container..."
  docker stop "$CONTAINER_NAME"
  docker rm "$CONTAINER_NAME"
else
  echo "No existing container to stop."
fi

# Build the Docker image
echo "Building the Docker image..."
docker build --platform "linux/amd64" -t "$DOCKER_IMAGE_NAME" .

# Run the container
echo "Deploying the application..."
docker run -d --name "$CONTAINER_NAME" --network "$NETWORK_NAME" -p "$EXPOSED_PORT:$EXPOSED_PORT" "$DOCKER_IMAGE_NAME"

echo "Deployment completed successfully!"
