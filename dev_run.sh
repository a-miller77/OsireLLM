#!/bin/bash

#SBATCH --job-name="OsireLLM-Dev"
#SBATCH --output=output/dev_%j.out
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

# Ensure output directory exists
mkdir -p output

# Set environment variables from .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Find an open port
find_port() {
    local start_port=8000
    local end_port=9000
    local port
    
    for ((port=start_port; port<=end_port; port++)); do
        if ! (echo > "/dev/tcp/localhost/$port") &>/dev/null; then
            echo "$port"
            return 0
        fi
    done
    
    echo "No open port found" >&2
    return 1
}

# Determine host and port
PORT=$(find_port)
HOST=$(hostname)
export BASE_URL="/node/${HOST}.hpc.msoe.edu/${PORT}"

echo "Starting development server on port $PORT"
echo "BASE_URL: $BASE_URL"

# Mount the local code directory into the container and run the API in development mode
singularity exec \
    --bind $(pwd)/app:/var/task/app \
    --bind $(pwd)/.env:/var/task/.env \
    --network-args portmap=$PORT:$PORT \
    ./image/container.sif \
    uvicorn --app-dir /var/task/app main:app --reload --port $PORT --host 0.0.0.0

echo "Development server stopped"