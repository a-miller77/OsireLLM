from typing import Dict
import asyncio
import socket
from fastapi import HTTPException
import logging
from datetime import datetime
import subprocess
import requests
from pathlib import Path
from core.models import JobStatus, JobState
from core.shell_commands import (
    run_async_scontrol,
    run_async_scancel,
    setup_ssh_keys_for_local_access
)
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

# Constants
ROSIE_WEB_DOMAIN = "https://dh-ood.hpc.msoe.edu"
ROSIE_BASE_URL_TEMPLATE = "{node_url}.hpc.msoe.edu:{port}"
DGX_MODEL_SERVER_URL = "http://dh-dgxh100-2.hpc.msoe.edu:8000"
DGX_MODEL_NAME = "meta/llama-3.1-70b-instruct"

# Port management
PREFERRED_PORTS = [
    8000,  # Standard web port
    7777,
    8080,
]
PREFERRED_PORTS.extend(x for x in range(8000, 8080))

def check_vllm_server(job_status: JobStatus) -> bool:
    """Check if vLLM server is healthy"""
    url = f"{job_status.server_url}/health"
    logger.debug(f"Checking vLLM server health at {url}")
    try:
        response = requests.get(url, timeout=2)
        is_healthy = response.status_code == 200
        logger.debug(f"vLLM server health check {'succeeded' if is_healthy else 'failed'}")
        return is_healthy
    except requests.RequestException as e:
        logger.debug(f"vLLM server health check failed: {str(e)}")
        return False

async def get_job_info(job_status: JobStatus) -> Dict:
    """Get detailed information about a SLURM job."""
    logger.debug(f"Getting info for job {job_status.job_id}")
    new_job_status = job_status.copy()
    try:
        # Get SLURM job info with async SSH command
        job_info, stderr, return_code = await run_async_scontrol(job_status.job_id)

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, 'scontrol', "", stderr)

        node = job_info.get('NodeList', None)
        status = JobState.UNKNOWN

        # Determine job status
        if job_info.get('JobState') == "RUNNING":
            new_job_status.node = node
            status = (JobState.RUNNING if node and check_vllm_server(new_job_status)
                     else JobState.STARTING)
        else:
            try:
                status = JobState(job_info.get('JobState', 'UNKNOWN'))
            except ValueError:
                logger.warning(f"Unknown SLURM status: {job_info.get('JobState')}")

        return {
            'status': status,
            'node': node,
            'error_message': job_info.get('Reason') if status == JobState.FAILED else None,
            'updated_at': datetime.utcnow()
        }

    except asyncio.TimeoutError:
        logger.error(f"Timeout getting info for job {job_status.job_id}")
        raise HTTPException(status_code=500, detail="SLURM command timed out")
    except Exception as e:
        logger.error(f"Failed to get job info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def cancel_slurm_job(job_status: JobStatus) -> None:
    """Execute SLURM job cancellation command."""
    logger.info(f"Executing cancellation for SLURM job {job_status.job_id}")
    try:
        stdout, stderr, return_code = await run_async_scancel(job_status.job_id)

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, 'scancel', stdout, stderr)

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to cancel SLURM job: {e.stderr}")
        raise HTTPException(status_code=500,
                          detail=f"Failed to terminate job: {e.stderr}")

class JobStateManager:
    """Manages the state of vLLM jobs with thread-safe operations"""
    def __init__(self):
        self._jobs: Dict[str, JobStatus] = {}
        self._dict_lock = asyncio.Lock()  # key reads, additions, deletions
        self._model_locks = defaultdict(asyncio.Lock)  # model-level updates
        self._cleanup_interval = 900  # 15 minutes in seconds
        self._update_interval = 300  # 5 minutes in seconds
        self._fast_update_interval = 3  # 3 seconds for new models (until RUNNING or FAILED)
        self._fast_update_backoff = 30  # poll every 30 seconds after 5 minutes
        self._fast_update_duration = 300  # Run fast updates for 5 minutes (300 seconds)
        self._cleanup_task = None
        self._update_task = None
        self._fast_update_tasks = {}  # Track fast update tasks by model name
        self._shutdown_event = asyncio.Event()

        # Initialize SSH setup task
        self._ssh_setup_task = asyncio.create_task(self._setup_ssh())

        # Check for DGX model and register it if available
        self._dgx_check_task = asyncio.create_task(self._check_dgx_model())

    async def _setup_ssh(self):
        """Set up SSH for local connections."""
        logger.info("Setting up SSH for local connections")
        try:
            # Directly await the async function instead of using to_thread
            success, message = await setup_ssh_keys_for_local_access(
                key_name="id_rsa_osire"
            )
            if success:
                logger.info(f"SSH setup successful: {message}")
            else:
                logger.warning(f"SSH setup issue: {message}")
        except Exception as e:
            logger.error(f"Error setting up SSH: {str(e)}")

    async def _check_dgx_model(self):
        """Check if the DGX model server is available and register it if so."""
        logger.info(f"Checking for DGX model server at {DGX_MODEL_SERVER_URL}")
        try:
            # Create a temporary JobStatus to use with check_vllm_server
            temp_status = JobStatus(
                job_id="dgx-static",
                model_name=DGX_MODEL_NAME,
                num_gpus=4,
                partition="dgxh100",
                status=JobState.UNKNOWN,
                server_url=DGX_MODEL_SERVER_URL,
                port=8000,
                node="dh-dgxh100-2",
                created_at=datetime.utcnow(),
                updated_at=None
            )

            # Check if the server is responding
            if requests.get(f"{DGX_MODEL_SERVER_URL}/v1/health/ready").status_code == 200:
                logger.info(f"DGX model server is active at {DGX_MODEL_SERVER_URL}")
                # Update status to RUNNING and add to job dictionary
                temp_status.status = JobState.RUNNING

                async with self._dict_lock:
                    self._jobs[DGX_MODEL_NAME] = temp_status
                    logger.info(f"Registered DGX model as {DGX_MODEL_NAME}")
            else:
                logger.info(f"DGX model server at {DGX_MODEL_SERVER_URL} is not active")
        except Exception as e:
            logger.error(f"Error checking DGX model server: {str(e)}")

    async def acquire_port(self) -> int:
        """Get port for a new job, trying preferred ports in order"""
        logger.info("Attempting to acquire port from preferred list")

        port_list = PREFERRED_PORTS
        for port in port_list:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    logger.info(f"Successfully acquired port: {port}")
                    return port
            except OSError:
                logger.debug(f"Port {port} unavailable, trying next")
                if port == port_list[-1]:
                    logger.error("No ports available")
                    raise HTTPException(status_code=503, detail="No ports available")

    async def add_job(self, model_name: str, job_status: JobStatus) -> None:
        """Add a new job to the state manager"""
        async with self._dict_lock:
            if model_name in self._jobs and self._jobs[model_name].status.is_active:
                logger.warning(f"Model {model_name} is already running")
                raise HTTPException(
                    status_code=409,
                    detail=f"Model {model_name} is already running"
                )
            self._jobs[model_name] = job_status
            logger.info(f"Successfully added job for model {model_name}")

            # Start fast updates for the new model
            self._start_fast_updates_for_model(model_name)

    async def remove_job(self, model_name: str) -> None:
        """Remove a job from the state manager"""
        async with self._dict_lock:
            logger.info(f"Removing job for model {model_name}")
            if model_name not in self._jobs:
                logger.warning(f"No job found for model {model_name}")
                raise HTTPException(
                    status_code=404,
                    detail=f"No job found for model {model_name}"
                )
            del self._jobs[model_name]
            logger.info(f"Successfully removed job for model {model_name}")

    async def get_job(self, model_name: str) -> JobStatus:
        """Get a specific job's status"""
        async with self._dict_lock:
            if model_name not in self._jobs:
                raise HTTPException(
                    status_code=404,
                    detail=f"No active job found for model {model_name}"
                )
            return self._jobs[model_name]

    async def update_job_status(self, job_status: JobStatus) -> JobStatus:
        """Update the status of a job with per-model locking"""
        model_name = job_status.model_name
        logger.debug(f"Updating status for job {job_status.job_id} (model {model_name})")

        # Use the model-specific lock for updates
        async with self._model_locks[model_name]:
            try:
                # Special handling for non-SLURM models (like DGX static models)
                if not job_status.job_id.isdigit():
                    # For non-SLURM models, just check if the server is healthy
                    logger.debug(f"Checking health of non-SLURM model {model_name}")
                    is_healthy = requests.get(f"{DGX_MODEL_SERVER_URL}/v1/health/ready").status_code == 200
                    job_status.status = JobState.RUNNING if is_healthy else JobState.FAILED
                    job_status.updated_at = datetime.utcnow()
                    job_status.error_message = None if is_healthy else "Server health check failed"
                    return job_status

                # Regular SLURM model update logic
                job_info = await get_job_info(job_status)

                # Update fields that can change
                job_status.status = job_info['status']
                job_status.updated_at = job_info['updated_at']
                if job_info['node']:
                    job_status.node = job_info['node']
                if job_info['error_message']:
                    job_status.error_message = job_info['error_message']

                return job_status
            except Exception as e:
                logger.info(f"Failed to get job info, marking as UNKNOWN: {str(e)}")
                job_status.status = JobState.UNKNOWN
                job_status.error_message = str(e)
                job_status.updated_at = datetime.utcnow()
                return job_status

    async def get_all_jobs(self) -> Dict[str, JobStatus]:
        """Get all jobs and update their status"""
        # Take a snapshot of jobs first
        job_snapshot = []
        async with self._dict_lock:
            job_snapshot = [(model_name, job_status.copy())
                            for model_name, job_status
                            in self._jobs.items()]

        logger.debug(f"Getting status for {len(job_snapshot)} jobs")

        # Update all jobs concurrently using model-level locks
        update_tasks = []
        for model_name, job_status in job_snapshot:
            update_tasks.append(self._update_single_job(model_name, job_status))

        if update_tasks:
            await asyncio.gather(*update_tasks, return_exceptions=True)

        # Get final snapshot for return
        async with self._dict_lock:
            current_jobs = self._jobs.copy()

            return current_jobs

    async def is_model_running(self, model_name: str) -> bool:
        """Check if a specific model is currently running"""
        async with self._dict_lock:
            if model_name not in self._jobs:
                return False
            return self._jobs[model_name].status.is_active

    async def start_cleanup_task(self):
        """Start the periodic cleanup task"""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._run_cleanup_loop())
            logger.info("Started periodic job cleanup task")

        # Also start the update task if not already running
        if not self._update_task:
            self._update_task = asyncio.create_task(self._run_update_loop())
            logger.info("Started periodic job update task")

    async def stop_cleanup_task(self):
        """Stop the cleanup task"""
        if self._cleanup_task:
            self._shutdown_event.set()
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        # Also stop the update task
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None

        # Stop any fast update tasks
        for task in self._fast_update_tasks.values():
            if not task.done():
                task.cancel()
        self._fast_update_tasks.clear()

        self._shutdown_event.clear()
        logger.info("Stopped periodic tasks")

    async def _run_cleanup_loop(self):
        """Run the cleanup loop every cleanup_interval seconds"""
        await asyncio.sleep(self._cleanup_interval)
        while not self._shutdown_event.is_set():
            try:
                # First update all jobs, then clean up based on updated status
                await self._update_all_jobs()
                await self.cleanup_jobs()
                await asyncio.sleep(self._cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(self._cleanup_interval)

    async def cleanup_jobs(self):
        """Clean up finished jobs"""
        # First identify jobs that need to be cleaned up
        jobs_to_cleanup = []

        async with self._dict_lock:
            for model_name, job_status in list(self._jobs.items()):
                if job_status.status.is_finished:
                    jobs_to_cleanup.append(model_name)

        # If we have jobs to clean up, acquire lock once and remove them all
        if jobs_to_cleanup:
            async with self._dict_lock:
                for model_name in jobs_to_cleanup:
                    if model_name in self._jobs and self._jobs[model_name].status.is_finished:
                        del self._jobs[model_name]
                        logger.info(f"Cleaned up finished job for model {model_name}")

    async def terminate_job(self, model_name: str) -> None:
        """Terminate a job with proper locking."""
        logger.info(f"Terminating job for model {model_name}")

        job_status = None
        async with self._dict_lock:
            job_status = self._jobs[model_name]
        try:
            async with self._model_locks[model_name]:
                await cancel_slurm_job(job_status)

                # Update job status to reflect termination
                job_status.status = JobState.CANCELLED
                job_status.updated_at = datetime.utcnow()
                if model_name in self._jobs:
                    self._jobs[model_name] = job_status
        except Exception as e:
            logger.error(f"Failed to terminate job for model {model_name}: {str(e)}")
        logger.info(f"Successfully terminated job for model {model_name}")

    async def cancel_all_jobs(self) -> None:
        """Cancel all active jobs during shutdown"""
        logger.info("Cancelling all active jobs")
        async with self._dict_lock:
            for model_name, job_status in list(self._jobs.items()):
                if job_status.status.is_active:
                    try:
                        # Use the model lock for each cancellation
                        async with self._model_locks[model_name]:
                            await cancel_slurm_job(job_status)
                        logger.info(f"Successfully cancelled job for model {model_name}")
                    except Exception as e:
                        logger.error(f"Failed to cancel job for model {model_name}: {str(e)}")

    async def _run_update_loop(self):
        """Run the update loop every update_interval seconds"""
        await asyncio.sleep(self._update_interval)
        while not self._shutdown_event.is_set():
            try:
                # Only update if it's not time for a cleanup
                # Calculate time since last cleanup
                current_time = asyncio.get_event_loop().time()
                time_since_cleanup = current_time % self._cleanup_interval

                # Calculate time until next cleanup (inverted from time_since_cleanup)
                time_until_next_cleanup = self._cleanup_interval - time_since_cleanup

                # Skip updates that would happen right before cleanup
                if time_until_next_cleanup < self._update_interval * 0.2:
                    logger.debug(f"Skipping update as cleanup will run soon (in {time_until_next_cleanup:.1f}s)")
                else:
                    await self._update_all_jobs()

                await asyncio.sleep(self._update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
                await asyncio.sleep(self._update_interval)

    async def _update_all_jobs(self):
        """Update status for all jobs"""
        # Take a snapshot of jobs to update while holding global lock
        job_snapshot = []
        async with self._dict_lock:
            job_snapshot = [(model_name, job_status.copy()) for model_name, job_status in self._jobs.items()]

        logger.debug(f"Updating status for {len(job_snapshot)} jobs")

        # Process each job update concurrently using model-level locks
        update_tasks = []
        for model_name, job_status in job_snapshot:
            update_tasks.append(self._update_single_job(model_name, job_status))

        # Run all updates concurrently
        if update_tasks:
            await asyncio.gather(*update_tasks, return_exceptions=True)

    async def _update_single_job(self, model_name: str, job_status: JobStatus):
        """Update a single job with proper locking"""
        try:
            # First update the status (this uses model-level locking internally)
            updated_status = await self.update_job_status(job_status)

            # Then update the shared dictionary with global lock
            async with self._dict_lock:
                if model_name in self._jobs:
                    self._jobs[model_name] = updated_status
                    logger.debug(f"Updated status for {model_name}: {updated_status.status}")
        except Exception as e:
            logger.error(f"Failed to update job {model_name}: {str(e)}")

    def _start_fast_updates_for_model(self, model_name: str):
        """Start a fast update task for a newly created model"""
        # Cancel any existing task for this model
        if model_name in self._fast_update_tasks and not self._fast_update_tasks[model_name].done():
            self._fast_update_tasks[model_name].cancel()

        # Create a new task
        self._fast_update_tasks[model_name] = asyncio.create_task(
            self._run_fast_updates_for_model(model_name)
        )
        logger.info(f"Started fast updates for model {model_name}")

    async def _run_fast_updates_for_model(self, model_name: str):
        """Run frequent updates for a specific model until it reaches RUNNING state or fails"""
        iterations = 0
        start_time = asyncio.get_event_loop().time()
        current_interval = self._fast_update_interval

        try:
            while True:
                iterations += 1
                logger.debug(f"Fast update #{iterations} for model {model_name}")

                # Check if we should switch to backoff interval
                elapsed_time = asyncio.get_event_loop().time() - start_time
                if elapsed_time > self._fast_update_duration and current_interval == self._fast_update_interval:
                    logger.info(f"Switching to backoff interval for model {model_name} after {elapsed_time:.1f}s")
                    current_interval = self._fast_update_backoff

                async with self._dict_lock:
                    if model_name not in self._jobs:
                        logger.info(f"Model {model_name} no longer exists, stopping fast updates")
                        break

                    job_status = self._jobs[model_name]

                    # If the job has reached a terminal state or is now running, stop updates
                    if job_status.status.is_finished:
                        logger.info(f"Model {model_name} reached terminal state {job_status.status}, stopping fast updates")
                        break
                    elif job_status.status == JobState.RUNNING:
                        logger.info(f"Model {model_name} is now running, stopping fast updates")
                        break

                    # Update the job status
                    try:
                        updated_status = await self.update_job_status(job_status)
                        self._jobs[model_name] = updated_status
                    except Exception as e:
                        logger.error(f"Failed to fast update job {model_name}: {str(e)}")

                await asyncio.sleep(current_interval)

            logger.info(f"Completed fast updates for model {model_name} after {iterations} iterations")
        except asyncio.CancelledError:
            logger.info(f"Fast updates for model {model_name} were cancelled")
        except Exception as e:
            logger.error(f"Error in fast updates for model {model_name}: {str(e)}")
        finally:
            # Clean up the task reference
            if model_name in self._fast_update_tasks:
                del self._fast_update_tasks[model_name]

# Create a global instance
resource_manager = JobStateManager()
#TODO run setup commands

# Dependency for FastAPI
async def get_resource_manager() -> JobStateManager:
    return resource_manager