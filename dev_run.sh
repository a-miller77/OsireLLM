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
# Set -e flag to exit immediately if a command exits with a non-zero status
set -e

# Use timeout to kill the process if it hangs
    # --bind /usr/bin/sbatch:/usr/bin/sbatch \
    # --bind /usr/lib/x86_64-linux-gnu/slurm:/usr/lib/x86_64-linux-gnu/slurm \
    # --bind /etc/slurm:/etc/slurm \
timeout --preserve-status 105m singularity exec \
    --bind $(pwd)/app:/var/task/app \
    --bind $(pwd)/.env:/var/task/.env \
    --bind /data:/data \
    --network-args portmap=$PORT:$PORT \
    ./osire-llm.sif \
    uvicorn --app-dir /var/task/app main:app --reload --port $PORT --host 0.0.0.0 \
    --reload-dir /var/task/app --reload-include "*.py"

# Capture the exit status
EXIT_STATUS=$?

# Check if the command timed out or failed
if [ $EXIT_STATUS -eq 124 ]; then
    echo "Development server timed out after 1h 45m"
elif [ $EXIT_STATUS -ne 0 ]; then
    echo "Development server crashed with exit code $EXIT_STATUS"
    # Force job termination on error
    scancel $SLURM_JOB_ID
    exit $EXIT_STATUS
else
    echo "Development server stopped normally"
fi

# Ensure the job terminates
exit $EXIT_STATUS