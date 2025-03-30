from typing import Dict, Optional, List
import asyncio
import socket
from fastapi import HTTPException
import logging
from datetime import datetime
from collections import defaultdict
from core.models import JobStatus, JobState, LaunchRequest
from core.settings import get_settings
from core.shell_commands import setup_ssh_keys_for_local_access
from .engines import factory as engine_factory
import copy

# Configure logging
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Port management from settings
PREFERRED_PORTS = settings.slurm.preferred_ports.copy()
PREFERRED_PORTS.extend(x for x in range(8001, 8080))

class JobStateManager:
    """Manages the state of inference engine jobs (Slurm, static, etc.)."""
    def __init__(self):
        self._jobs: Dict[str, JobStatus] = {}
        self._dict_lock = asyncio.Lock()  # Protects dictionary structure (keys, iteration, add/remove)
        self._model_locks = defaultdict(asyncio.Lock)  # Protects operations on specific job values

        # Background task intervals from settings
        self._cleanup_interval = settings.job_state_manager.cleanup_interval
        self._update_interval = settings.job_state_manager.update_interval
        self._fast_update_interval = settings.job_state_manager.fast_update_interval
        self._fast_update_backoff = settings.job_state_manager.fast_update_backoff
        self._fast_update_duration = settings.job_state_manager.fast_update_duration

        self._cleanup_task: Optional[asyncio.Task] = None
        self._update_task: Optional[asyncio.Task] = None
        self._fast_update_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()

        self._ssh_setup_task = asyncio.create_task(self._setup_ssh())
        self._static_model_check_task = asyncio.create_task(self._register_static_models())

    async def _setup_ssh(self):
        """Set up SSH for local connections."""
        logger.info("Setting up SSH for local connections")
        try:
            # Use settings for key_name and key_dir
            success, message = await setup_ssh_keys_for_local_access(
                key_name=settings.ssh.key_name,
                key_dir=settings.ssh.key_dir
            )
            if success:
                logger.info(f"SSH setup successful: {message}")
            else:
                logger.warning(f"SSH setup issue: {message}")
        except Exception as e:
            logger.error(f"Error setting up SSH: {str(e)}")

    async def _register_static_models(self):
        """Register and perform initial status check for static models from config."""
        logger.info("Registering static model servers defined in configuration...")
        static_job_statuses: List[JobStatus] = []
        for model_config in settings.static_models:
            logger.info(f"Preparing registration for static model '{model_config.model_name}' (ID: {model_config.id}) at {model_config.server_url}")
            # Create a preliminary JobStatus object
            # Determine engine type (should be added to StaticModelConfig in settings.py)
            engine_type = getattr(model_config, 'engine_type', 'unknown') # Default to unknown if missing
            if engine_type == 'unknown':
                 logger.error(f"Static model config for {model_config.id} is missing 'engine_type'. Skipping registration.")
                 continue

            temp_job_status = JobStatus(
                job_id=model_config.id,
                model_name=model_config.model_name,
                engine_type=engine_type,
                num_gpus=getattr(model_config, 'num_gpus', 0),
                partition=getattr(model_config, 'partition', 'static'),
                status=JobState.UNKNOWN, # Initial status before check
                server_url=model_config.server_url,
                port=getattr(model_config, 'port', None),
                node=getattr(model_config, 'node', 'static'),
                created_at=datetime.utcnow(),
                is_static=True
            )
            static_job_statuses.append(temp_job_status)

        # Perform initial status check concurrently using the refactored update_job_status
        update_tasks = []
        for temp_status in static_job_statuses:
            update_tasks.append(self.update_job_status(temp_status.copy())) # Pass a copy

        results = await asyncio.gather(*update_tasks, return_exceptions=True)

        # Add successfully checked static jobs to the main dictionary using dict_lock
        async with self._dict_lock: # CORRECT: Protects dictionary write
            for i, result in enumerate(results):
                temp_status = static_job_statuses[i] # Original status for logging errors
                if isinstance(result, JobStatus):
                    # Use the updated status returned by update_job_status
                    self._jobs[temp_status.model_name] = temp_status
                    logger.info(f"Registered static model '{temp_status.model_name}' (ID: {temp_status.job_id}) with status {temp_status.status}")
                elif isinstance(result, Exception):
                    logger.error(f"Initial status check failed for static model '{temp_status.model_name}': {result}. Not registering.")
                else:
                     logger.error(f"Unexpected result type {type(result)} during static model '{temp_status.model_name}' registration.")

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
        """Add a new job to the state manager and start fast updates."""
        async with self._dict_lock: # CORRECT: Protects dictionary write
            # We might replace an old finished job here if relaunching
            self._jobs[model_name] = job_status.copy() # Store a copy
            logger.info(f"Added/Updated job for model {model_name} with job_id {job_status.job_id}")
        # Start fast updates for the new job if it's not static
        if not job_status.is_static:
             self._start_fast_updates_for_model(model_name)

    async def remove_job(self, model_name: str) -> None:
        """Remove a job from the state manager (only non-static allowed)."""
        async with self._dict_lock: # CORRECT: Protects key check and removal
            logger.info(f"Removing job for model {model_name}")
            job_status = self._jobs.get(model_name)
            if not job_status:
                raise HTTPException(status_code=404, detail=f"No job found for model {model_name}")
            if job_status.is_static:
                raise HTTPException(status_code=403, detail=f"Static model '{model_name}' cannot be removed.")
            # Stop fast updates if running
            if model_name in self._fast_update_tasks:
                 self._fast_update_tasks[model_name].cancel()
                 # Best effort cancel, remove task entry regardless
                 del self._fast_update_tasks[model_name]
                 logger.debug(f"Cancelled fast update task for removed job {model_name}")
            # Remove from main dict
            del self._jobs[model_name]
            logger.info(f"Successfully removed job for model {model_name}")

    async def get_job(self, model_name: str) -> JobStatus:
        """Get a specific job's status."""
        async with self._dict_lock: # CORRECT: Protects dictionary read
            job_status = self._jobs.get(model_name)
            if not job_status:
                raise HTTPException(
                    status_code=404,
                    detail=f"No active job found for model {model_name}"
                )
            # UPDATED: Return a copy to prevent modification outside the lock
            return job_status.copy()

    async def update_job_status(self, job_status_input: JobStatus) -> JobStatus:
        """Update the status of a job using the appropriate engine instance, using two-phase locking."""
        model_name = job_status_input.model_name
        job_id = job_status_input.job_id
        logger.debug(f"Updating status for job {job_id} (model {model_name}) via engine '{job_status_input.engine_type}'")

        processed_status: Optional[JobStatus] = None

        # --- Phase 1: Process update using model lock ---
        try:
            async with self._model_locks[model_name]: # CORRECT: Protects processing of this specific job
                # Get the engine instance responsible for this job
                # Pass a copy so engine modifies its own version initially
                engine = engine_factory.get_manager_instance(job_status_input.copy())

                # Delegate status check to the engine
                # The engine's get_status method updates its internal job_status and returns it
                processed_status = await engine.get_status()

                # Update node URL based on processed status if needed
                # Check against the input status for changes
                if (processed_status.node != job_status_input.node or not processed_status.server_url) \
                   and processed_status.node and not processed_status.is_static:
                     self._update_server_url(processed_status) # Modifies processed_status in place

        except (ValueError, NotImplementedError) as e: # Errors from factory or engine stubs
            logger.error(f"Engine error updating status for job {job_id}: {e}")
            # Prepare status to indicate failure, based on input status
            processed_status = job_status_input.copy()
            processed_status.status = JobState.UNKNOWN
            processed_status.error_message = f"Engine error: {e}"
            processed_status.updated_at = datetime.utcnow()
        except Exception as e:
            logger.error(f"Unexpected error during engine status check for job {job_id}: {e}", exc_info=True)
            # Prepare status to indicate failure
            processed_status = job_status_input.copy()
            processed_status.status = JobState.UNKNOWN
            processed_status.error_message = f"Unexpected update error: {e}"
            processed_status.updated_at = datetime.utcnow()
        # --- End Phase 1 ---

        # Ensure we have a status object to work with, even if phase 1 failed unexpectedly
        if processed_status is None:
             logger.error(f"Status processing failed unexpectedly for job {job_id}. Setting to UNKNOWN.")
             processed_status = job_status_input.copy()
             processed_status.status = JobState.UNKNOWN
             processed_status.error_message = "Unexpected status processing failure"
             processed_status.updated_at = datetime.utcnow()


        # --- Phase 2: Commit update using dict lock ---
        try:
            async with self._dict_lock: # CORRECT: Protects dictionary write
                 # Check if job still exists (might have been removed concurrently)
                 # Check ID match in case model was replaced
                 current_job_in_dict = self._jobs.get(model_name)
                 if current_job_in_dict and current_job_in_dict.job_id == job_id:
                      # Commit the final status (make a copy for safety)
                      self._jobs[model_name] = processed_status.copy()
                      logger.debug(f"Committed updated status for job {job_id} to {processed_status.status}")
                 else:
                      logger.warning(f"Job {job_id} for model {model_name} was removed or replaced during status update. Discarding update commit.")
                      # Return the status determined by the processing phase, even if not committed
                      return processed_status

            # Return the committed status (or the processed one if commit was skipped)
            return processed_status

        except Exception as e:
             logger.error(f"Unexpected error committing status update for job {job_id}: {e}", exc_info=True)
             # Return the status determined by the processing phase, even if commit failed
             return processed_status
        # --- End Phase 2 ---

    async def terminate_job(self, model_name: str) -> None:
        """Terminate a specific job using the appropriate engine instance. Uses model lock only."""
        logger.info(f"Attempting termination for model {model_name}")

        # Initial read requires dict_lock to safely get the job state
        try:
             async with self._dict_lock: # CORRECT: Protects dictionary read
                  job_status_copy = self._jobs[model_name].copy() # Get a copy safely
        except KeyError:
             logger.warning(f"Attempted to terminate non-existent job for model '{model_name}'")
             raise HTTPException(status_code=404, detail=f"No job found for model {model_name}")

        # Check status based on the copy
        if job_status_copy.is_static:
            logger.warning(f"Attempted to terminate static model '{model_name}'. Operation not allowed.")
            raise HTTPException(status_code=403, detail=f"Cannot terminate static model '{model_name}'.")

        if job_status_copy.status.is_finished:
             logger.info(f"Job for model {model_name} (ID: {job_status_copy.job_id}) is already finished ({job_status_copy.status}). No termination needed.")
             return # Nothing to do

        # Perform termination action using model lock
        termination_success = False
        try:
            async with self._model_locks[model_name]: # CORRECT: Protects termination action for this job
                # Get engine instance using the job state copy
                engine = engine_factory.get_manager_instance(job_status_copy)
                # Attempt termination
                termination_success = await engine.terminate()

        except NotImplementedError as e:
            logger.error(f"Termination failed: Engine type '{job_status_copy.engine_type}' does not support termination: {e}")
            raise HTTPException(status_code=501, detail=f"Engine '{job_status_copy.engine_type}' does not support termination.")
        except HTTPException as he: # Re-raise 404 etc. from initial read if needed
             raise he
        except Exception as e:
            logger.error(f"Unexpected error during job termination action for model '{model_name}': {e}", exc_info=True)
            # Raise 500 for unexpected errors during the termination process itself
            raise HTTPException(status_code=500, detail=f"Unexpected error terminating job: {e}")

        if termination_success:
            logger.info(f"Termination command issued successfully for job {job_status_copy.job_id} (model {model_name}). Status will update in background.")
        else:
             # Log the failure, the background update will eventually reflect the state if it failed gracefully
             logger.error(f"Termination command failed or reported failure for job {job_status_copy.job_id} (model {model_name})")
             # Do we raise an exception here? Let's stick to raising 500 if the command itself fails.
             # If the command runs but returns False, it might just mean the job was already gone.
             # Let's not raise HTTPException here, just log. The job state will resolve later.

    async def get_all_jobs(self) -> Dict[str, JobStatus]:
        """Get status of all managed jobs"""
        async with self._dict_lock: # CORRECT: Protects dictionary iteration/copying
            # Return a deep copy to prevent modification outside the manager
            return copy.deepcopy(self._jobs)

    async def is_model_running(self, model_name: str) -> bool:
        """Check if a specific model is currently running"""
        async with self._dict_lock:
            if model_name not in self._jobs:
                return False
            return self._jobs[model_name].status.is_active

    async def start_cleanup_task(self):
        """Start the periodic cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._run_cleanup_loop())
            logger.info(f"Cleanup task started. Interval: {self._cleanup_interval}s")

    async def stop_cleanup_task(self):
        """Stop the cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                logger.info("Cleanup task stopped.")
            self._cleanup_task = None

    async def _run_cleanup_loop(self):
        """Periodically remove finished jobs from the manager."""
        await asyncio.sleep(self._cleanup_interval) # Initial delay
        while not self._shutdown_event.is_set():
            try:
                 logger.debug("Running cleanup task...")
                 jobs_to_remove = []
                 async with self._dict_lock: # CORRECT: Protects iteration and reads
                      current_time = datetime.utcnow()
                      for model_name, job_status in self._jobs.items():
                          # Only remove finished non-static jobs
                          if not job_status.is_static and job_status.status.is_finished:
                               # Optional: Add a grace period after finish time?
                               # if (current_time - job_status.updated_at).total_seconds() > 60:
                               jobs_to_remove.append(model_name)

                 # Remove jobs outside the initial lock to avoid holding it too long
                 # Each removal requires the dict_lock again via remove_job
                 if jobs_to_remove:
                      logger.info(f"Cleaning up {len(jobs_to_remove)} finished job(s): {', '.join(jobs_to_remove)}")
                      for model_name in jobs_to_remove:
                           try:
                                # remove_job handles its own dict_lock
                                await self.remove_job(model_name)
                           except HTTPException as e: # e.g., 404 if already removed
                                logger.warning(f"Issue during cleanup removal of {model_name}: {e.detail}")
                           except Exception as e:
                                logger.error(f"Unexpected error cleaning up job {model_name}: {e}", exc_info=True)

                 await asyncio.sleep(self._cleanup_interval)
            except asyncio.CancelledError:
                 logger.info("Cleanup loop cancelled.")
                 break
            except Exception as e:
                 logger.error(f"Error in cleanup loop: {e}", exc_info=True)
                 await asyncio.sleep(self._cleanup_interval) # Wait before retrying

    async def start_update_task(self):
        if self._update_task is None:
            self._update_task = asyncio.create_task(self._run_update_loop())
            logger.info(f"Background update task started. Interval: {self._update_interval}s")

    async def stop_update_task(self):
         if self._update_task:
             self._update_task.cancel()
             try:
                 await self._update_task
             except asyncio.CancelledError:
                 logger.info("Background update task stopped.")
             self._update_task = None

    async def _run_update_loop(self):
        """Periodically update the status of all managed jobs not in fast-update mode."""
        await asyncio.sleep(5) # Give some time for init tasks
        logger.info("Starting background job update loop...")
        while not self._shutdown_event.is_set():
            try:
                logger.debug("Running scheduled job status update scan...")
                await self._update_all_monitored_jobs() # Helper performs the work

                await asyncio.sleep(self._update_interval)
            except asyncio.CancelledError:
                logger.info("Background update loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in background update loop: {e}", exc_info=True)
                await asyncio.sleep(self._update_interval) # Wait before retrying

    async def _run_fast_update_loop(self, model_name: str):
        """Run fast updates for a specific job for a limited duration."""
        start_time = asyncio.get_event_loop().time()
        backoff_time = self._fast_update_interval
        logger.info(f"Starting fast update loop for model '{model_name}'...")
        job_removed = False

        while (asyncio.get_event_loop().time() - start_time) < self._fast_update_duration:
            if self._shutdown_event.is_set():
                break # Exit if shutdown is triggered

            try:
                # Get current status first using get_job (which uses dict_lock and returns copy)
                job_status_copy = await self.get_job(model_name)

                if job_status_copy.status.is_finished:
                     logger.info(f"Stopping fast updates for '{model_name}' as job is finished.")
                     break # Stop fast updates if job is done

                logger.debug(f"Running fast update for '{model_name}'...")
                # Directly call the refactored update_job_status
                # update_job_status handles its own locking
                updated_status = await self.update_job_status(job_status_copy)

                # Check status *after* update
                if updated_status.status.is_finished:
                      logger.info(f"Stopping fast updates for '{model_name}' as job reached finished state after update.")
                      break

                await asyncio.sleep(backoff_time)
                backoff_time = min(backoff_time * self._fast_update_backoff, self._update_interval) # Exponential backoff

            except asyncio.CancelledError:
                logger.info(f"Fast update loop for '{model_name}' cancelled.")
                break
            except HTTPException as e:
                 if e.status_code == 404:
                      logger.info(f"Stopping fast updates for '{model_name}' as job was not found (likely removed).")
                      job_removed = True # Mark as removed to skip final cleanup
                      break
                 else:
                      logger.error(f"HTTP error during fast update for '{model_name}': {e.detail}", exc_info=True)
                      await asyncio.sleep(backoff_time) # Wait before retrying
            except Exception as e:
                logger.error(f"Error in fast update loop for '{model_name}': {e}", exc_info=True)
                await asyncio.sleep(backoff_time) # Wait before retrying

        logger.info(f"Fast update loop for model '{model_name}' finished.")
        if not job_removed:
             async with self._dict_lock: # CORRECT: Protects modification of _fast_update_tasks
                  if model_name in self._fast_update_tasks:
                       del self._fast_update_tasks[model_name]

    async def _update_all_monitored_jobs(self):
        """Helper function to update status for all jobs not in fast-update mode."""
        job_snapshot: List[JobStatus] = []
        async with self._dict_lock: # CORRECT: Protects dictionary iteration/read
            # Get jobs that are NOT currently handled by a fast update task AND are not finished
            jobs_to_update = {
                 name: status for name, status in self._jobs.items()
                 if name not in self._fast_update_tasks and not status.status.is_finished
            }
            if not jobs_to_update:
                 # logger.debug("No jobs require regular background update.") # Too verbose
                 return

            logger.debug(f"Updating status for {len(jobs_to_update)} jobs via regular background task.")
            # Create copies for safe concurrent processing
            job_snapshot = [status.copy() for status in jobs_to_update.values()]

        # Create update tasks for each job
        update_tasks = []
        for job_status_copy in job_snapshot:
             # Call the refactored update_job_status method for each job copy
             # update_job_status handles its own locking internally
             update_tasks.append(self.update_job_status(job_status_copy))

        # Run updates concurrently and log any errors
        results = await asyncio.gather(*update_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Log error, associate it back to the model name from the snapshot
                model_name = job_snapshot[i].model_name
                logger.error(f"Error updating job '{model_name}' during background scan: {result}", exc_info=isinstance(result, Exception) and result or False)

    def _start_fast_updates_for_model(self, model_name: str):
        """Start a fast update task for a newly created model, ensuring only one runs."""
        # No lock needed here initially, task creation itself is atomic enough.
        # The loop internally handles removal from the dict safely.
        if model_name not in self._fast_update_tasks:
            logger.info(f"Starting fast updates for new model '{model_name}'")
            task = asyncio.create_task(self._run_fast_update_loop(model_name))
            # Technically a small race window here if called twice rapidly before task assignment
            # but dict lock later prevents issues adding the task entry itself if needed.
            # Let's add dict lock for safety assigning to the dict.
            async def assign_task(): # Helper coroutine to use async with
                 async with self._dict_lock:
                     # Check again inside lock
                     if model_name not in self._fast_update_tasks:
                         self._fast_update_tasks[model_name] = task
                     else:
                         # Task already created by another coroutine, cancel this new one
                         task.cancel()
                         logger.warning(f"Race condition detected starting fast updates for {model_name}. Using existing task.")
            asyncio.create_task(assign_task())
        else:
            logger.debug(f"Fast updates already running for model '{model_name}'")

    def _update_server_url(self, job_status: JobStatus):
        """Helper to update the server_url based on node and port. Modifies object in place."""
        # This is called within _model_locks block in update_job_status
        if job_status.node and job_status.port and not job_status.is_static:
            try:
                # Construct the node URL part using the template from settings
                node_part = job_status.node.split('.')[0] # Ensure node name doesn't already contain domain
                node_url_part = settings.server.node_url_template.format(
                    node_url=node_part,
                    port=job_status.port
                )
                full_url = f"http://{node_url_part}" # Prepend http scheme
                if job_status.server_url != full_url:
                     logger.info(f"Updating server_url for job {job_status.job_id} from '{job_status.server_url}' to '{full_url}'")
                     job_status.server_url = full_url
            except Exception as e:
                 logger.error(f"Error formatting node_url_template for job {job_status.job_id}: {e}", exc_info=True)
                 job_status.server_url = None # Clear URL if formatting fails

    async def shutdown(self):
        """Gracefully shutdown the JobStateManager, stopping background tasks."""
        logger.info("Shutting down JobStateManager...")
        self._shutdown_event.set()

        # Stop background tasks
        await self.stop_update_task()
        await self.stop_cleanup_task()

        # Cancel any running fast update tasks
        tasks_to_cancel = []
        async with self._dict_lock: # Protect access to _fast_update_tasks
            tasks_to_cancel = list(self._fast_update_tasks.values())
            self._fast_update_tasks.clear()

        if tasks_to_cancel:
             logger.info(f"Cancelling {len(tasks_to_cancel)} fast update tasks...")
             for task in tasks_to_cancel:
                  task.cancel()
             await asyncio.gather(*tasks_to_cancel, return_exceptions=True) # Wait for cancellation
             logger.info("Fast update tasks cancelled.")

        # Optionally: Cancel all active SLURM jobs? (Requires care, might not be desired)
        # await self.cancel_all_jobs() # Requires implementing this method safely

        logger.info("JobStateManager shutdown complete.")

# Create a global instance
resource_manager = JobStateManager()
#TODO refactor and move setup commands here

# Dependency for FastAPI
async def get_resource_manager() -> JobStateManager:
    # Ensure background tasks are started when the manager is first retrieved
    # Use asyncio.call_soon_threadsafe or similar if needed in a threaded context,
    # but for FastAPI startup/dependency injection, this should be okay.
    if resource_manager._update_task is None:
         await resource_manager.start_update_task()
    if resource_manager._cleanup_task is None:
         await resource_manager.start_cleanup_task()
    return resource_manager