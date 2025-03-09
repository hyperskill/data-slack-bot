#!/bin/bash

# Build the Docker image
docker build -t slack_bot .

# Test connectivity to the Slack extractor URL
echo "Testing connectivity to Slack extractor URL..."
source ./.env
docker run --network=host --rm slack_bot bash -c "curl -v $SLACK_EXTRACTOR_BASE_URL" || echo "Failed to connect to Slack extractor URL"

# Check if container exists and remove it
if [ "$(docker ps -a -q -f name=slack_bot_container)" ]; then
    echo "Removing existing slack_bot_container..."
    docker rm -f slack_bot_container
fi

# Run the Docker container with environment variables and host network
docker run --network=host --env-file ./.env --name slack_bot_container --rm -it slack_bot
