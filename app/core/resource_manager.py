from typing import Dict
import asyncio
import socket
from fastapi import HTTPException
import logging
from datetime import datetime
from core.models import JobStatus, JobState
from core.osire_llm_service import get_job_info, check_vllm_server, terminate_job

# Configure logging
logger = logging.getLogger(__name__)

# Port management
PREFERRED_PORTS = [
    8000,  # Standard web port
    7777,
    8080,
]

class JobStateManager:
    """Manages the state of vLLM jobs with thread-safe operations"""
    def __init__(self):
        self._jobs: Dict[str, JobStatus] = {}
        self._job_lock = asyncio.Lock()  # For job dictionary operations
        self._cleanup_interval = 1800  # 30 minutes in seconds
        self._cleanup_task = None
        self._shutdown_event = asyncio.Event()
    
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
                if port == PREFERRED_PORTS[-1]:
                    port_list.extend(x for x in range(8000, 8080))
                continue
        
        logger.error("No ports available")
        raise HTTPException(status_code=503, detail="No ports available")

    async def add_job(self, model_name: str, job_status: JobStatus) -> None:
        """Add a new job to the state manager"""
        async with self._job_lock:
            if model_name in self._jobs and self._jobs[model_name].status.is_active:
                logger.warning(f"Model {model_name} is already running")
                raise HTTPException(
                    status_code=409,
                    detail=f"Model {model_name} is already running"
                )
            self._jobs[model_name] = job_status
            logger.info(f"Successfully added job for model {model_name}")
    
    async def remove_job(self, model_name: str) -> None:
        """Remove a job from the state manager"""
        async with self._job_lock:
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
        async with self._job_lock:
            if model_name not in self._jobs:
                raise HTTPException(
                    status_code=404,
                    detail=f"No active job found for model {model_name}"
                )
            return self._jobs[model_name]

    async def update_job_status(self, job_status: JobStatus) -> JobStatus:
        """Update the status of a job"""
        logger.debug(f"Updating status for job {job_status.job_id}")
        try:
            job_info = await get_job_info(job_status.job_id, job_status.port)
            
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
        async with self._job_lock:
            logger.debug("Getting status for all jobs")
            try:
                # Update all job statuses
                for model_name, job_status in list(self._jobs.items()):
                    try:
                        updated_status = await self.update_job_status(job_status)
                        self._jobs[model_name] = updated_status
                    except Exception as e:
                        logger.error(f"Failed to update job {model_name}: {str(e)}")
                
                # Get current jobs before cleanup
                current_jobs = self._jobs.copy()
                
                # Schedule cleanup without waiting
                asyncio.create_task(self.cleanup_jobs())
                
                return current_jobs
            except Exception as e:
                logger.error(f"Failed to get all jobs: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    async def is_model_running(self, model_name: str) -> bool:
        """Check if a specific model is currently running"""
        async with self._job_lock:
            if model_name not in self._jobs:
                return False
            return self._jobs[model_name].status.is_active

    async def start_cleanup_task(self):
        """Start the periodic cleanup task"""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._run_cleanup_loop())
            logger.info("Started periodic job cleanup task")

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
            self._shutdown_event.clear()
            logger.info("Stopped periodic job cleanup task")

    async def _run_cleanup_loop(self):
        """Run the cleanup loop every cleanup_interval seconds"""
        while not self._shutdown_event.is_set():
            try:
                await self.cleanup_jobs()
                await asyncio.sleep(self._cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(self._cleanup_interval)

    async def cleanup_jobs(self):
        """Clean up finished jobs"""
        async with self._job_lock:
            for model_name, job_status in list(self._jobs.items()):
                if job_status.status.is_finished:
                    del self._jobs[model_name]
                    logger.info(f"Cleaned up finished job for model {model_name}")

    async def cancel_all_jobs(self) -> None:
        """Cancel all active jobs during shutdown"""
        logger.info("Cancelling all active jobs")
        async with self._job_lock:
            for model_name, job_status in list(self._jobs.items()):
                if job_status.status.is_active:
                    try:
                        await terminate_job(job_status)
                        logger.info(f"Successfully cancelled job for model {model_name}")
                    except Exception as e:
                        logger.error(f"Failed to cancel job for model {model_name}: {str(e)}")
            self._jobs.clear()

# Create a global instance
resource_manager = JobStateManager()

# Dependency for FastAPI
async def get_resource_manager() -> JobStateManager:
    return resource_manager 