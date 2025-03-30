import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import os
from datetime import datetime
import requests
# Core imports
from ..models import JobStatus, VLLMConfig, SlurmConfig, JobState
from ..settings import get_settings

from ..shell_commands import run_async_sbatch, run_async_scancel, run_async_scontrol, run_async_sacct

from .base import InferenceEngine

logger = logging.getLogger(__name__)
settings = get_settings()

# Directory for saving sbatch scripts
SBATCH_DIR = Path(settings.slurm.output_dir)
SBATCH_DIR.mkdir(parents=True, exist_ok=True)

class VllmEngine(InferenceEngine):
    """
    InferenceEngine implementation for vLLM models launched via Slurm.
    Handles launch validation and execution via static methods,
    and job management (status, termination) via instance methods.
    """

    # --- Static Methods for Launch --- #

    #TODO: this might be able to be moved into the base class at a later date
    @staticmethod
    def validate_launch_args(args: Dict[str, Any]) -> Tuple[VLLMConfig, SlurmConfig]:
        """
        Validates the engine-specific arguments dictionary using VLLMConfig and SlurmConfig.

        Args:
            args: The dictionary potentially containing vLLM and Slurm parameters.

        Returns:
            A tuple containing validated (VLLMConfig, SlurmConfig) instances.

        Raises:
            ValueError: If validation fails based on Pydantic models.
            KeyError: If required keys are missing (handled by Pydantic).
        """
        logger.debug("Validating vLLM launch arguments")
        try:
            # Extract fields for VLLMConfig
            vllm_field_names = set(VLLMConfig.model_fields.keys()) - {'vllm_extra_args'}
            vllm_data = {k: v for k, v in args.items() if k in vllm_field_names}
            # Handle potential extra args dict
            vllm_extra_args_dict = args.get("vllm_extra_args")
            if vllm_extra_args_dict and isinstance(vllm_extra_args_dict, dict):
                 vllm_config = VLLMConfig(**vllm_data, vllm_extra_args=vllm_extra_args_dict)
            else:
                 vllm_config = VLLMConfig(**vllm_data)

            # Extract fields for SlurmConfig
            slurm_field_names = set(SlurmConfig.model_fields.keys()) - {'slurm_extra_args'}
            slurm_data = {k: v for k, v in args.items() if k in slurm_field_names}
            # Handle potential extra args dict
            slurm_extra_args_dict = args.get("slurm_extra_args")
            if slurm_extra_args_dict and isinstance(slurm_extra_args_dict, dict):
                 slurm_config = SlurmConfig(**slurm_data, slurm_extra_args=slurm_extra_args_dict)
            else:
                 slurm_config = SlurmConfig(**slurm_data)

            logger.debug(f"Validation successful. VLLMConfig: {vllm_config}, SlurmConfig: {slurm_config}")
            return vllm_config, slurm_config
        except Exception as e:
            logger.error(f"Launch argument validation failed: {e}", exc_info=True)
            raise ValueError(f"Invalid launch arguments for vLLM/Slurm engine: {e}") from e

    @staticmethod
    def generate_script(vllm_config: VLLMConfig, slurm_config: SlurmConfig, port: int, model_name: str) -> Tuple[str, str]:
        """
        Generate the Slurm sbatch script content for launching vLLM.
        Static method using validated config objects.

        Returns:
            Tuple[str, str]: (script_content, job_name)
        """
        logger.debug(f"Generating SLURM script for model {model_name} on port {port}")

        # Construct job name (ensure filesystem-friendly)
        safe_model_name = model_name.replace('/', '_')
        job_name = slurm_config.job_name or f"OsireLLM_{safe_model_name}_{port}"

        # Construct output file paths (relative to SBATCH_DIR)
        user = os.environ.get("USER", "unknown_user")
        stdout_file = slurm_config.output_config.stdout_file or str(SBATCH_DIR / f"{job_name}.stdout")
        stderr_file = slurm_config.output_config.stderr_file or str(SBATCH_DIR / f"{job_name}.stderr")
        container_image = slurm_config.container
        download_dir = vllm_config.download_dir

        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={slurm_config.partition}",
            f"#SBATCH --nodes={slurm_config.nodes}",
            f"#SBATCH --gpus={slurm_config.gpus}",
            f"#SBATCH --cpus-per-gpu={slurm_config.cpus_per_gpu}",
            f"#SBATCH --time={slurm_config.time_limit}",
            f"#SBATCH --output={stdout_file}",
            f"#SBATCH --error={stderr_file}",
        ]
        if slurm_config.slurm_extra_args:
            for key, value in slurm_config.slurm_extra_args.items():
                script_lines.append(f"#SBATCH --{key}={value}")

        script_lines.extend([
            "", "module load singularity", "", "echo \"Running on node: $SLURMD_NODENAME\"", "",
            (
                f"singularity exec --nv -B {','.join(slurm_config.container_mounts)} "
                f"{container_image} python3 -m vllm.entrypoints.openai.api_server \\"
            ),
            f"    --model {model_name} \\",
            f"    --download-dir {download_dir} \\",
            "    --host 0.0.0.0 \\",
            f"    --port {port} \\",
            f"    --max-num-batched-tokens {vllm_config.max_num_batched_tokens} \\",
            f"    --gpu-memory-utilization {vllm_config.gpu_memory_utilization} \\",
            f"    --dtype {vllm_config.dtype} \\",
            f"    --max-model-len {vllm_config.max_model_len} \\",
            f"    --tensor-parallel-size {slurm_config.gpus} \\", # Match SLURM GPU allocation
            f"    --pipeline-parallel-size {slurm_config.nodes} \\", # Match SLURM node allocation
            f"    --uvicorn-log-level {slurm_config.output_config.log_level}",
        ])

        # Add vLLM extra args
        if vllm_config.vllm_extra_args:
            for key, value in vllm_config.vllm_extra_args.items():
                script_lines.append(f"    --{key} {value} \\")
        # Remove trailing backslash if necessary
        if script_lines[-1].endswith(" \\"):
             script_lines[-1] = script_lines[-1][:-3]

        script_content = "\n".join(script_lines) + "\n"
        logger.debug(f"Generated script content for job '{job_name}':\n{script_content}")
        return script_content, job_name

    @staticmethod
    async def submit_launch_command(script_content: str, job_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Save the script and launch the vLLM job using sbatch via asyncssh.
        Static method.

        Returns:
            Tuple[Optional[str], Optional[str]]: (job_id, error_message)
        """
        script_path = SBATCH_DIR / f"{job_name}.sh"
        logger.debug(f"Saving SLURM script for job '{job_name}' to {script_path}")
        try:
            with open(script_path, 'w') as f: f.write(script_content)
        except IOError as e:
            logger.error(f"Failed to save job script {script_path}: {e}", exc_info=True)
            return None, f"Failed to save job script: {e}"

        logger.info(f"Submitting SLURM script: {script_path}")
        try:
            stdout, stderr, return_code = await run_async_sbatch(str(script_path))
            if return_code != 0:
                error_detail = stderr or f"sbatch exited with code {return_code}"
                logger.error(f"sbatch submission failed for script {script_path}. Error: {error_detail}")
                return None, f"sbatch failed: {error_detail}"
            try:
                job_id_str = stdout.strip().split()[-1]
                job_id = int(job_id_str)
                logger.info(f"sbatch submission successful for script {script_path}. Job ID: {job_id}")
                return str(job_id), None
            except (IndexError, ValueError) as e:
                logger.error(f"Failed to parse job ID from sbatch output '{stdout}' for script {script_path}: {e}", exc_info=True)
                return None, f"Invalid job ID received from sbatch: {stdout}"
        except Exception as e:
            logger.error(f"Exception during sbatch submission for script {script_path}: {e}", exc_info=True)
            return None, f"Failed to submit job: {e}"

    # --- Instance Methods for Management --- #

    def __init__(self, job_status: JobStatus):
        """
        Initialize the vLLM engine instance for managing an existing job.
        """
        super().__init__(job_status)

    async def _check_http_status(self, url: str, endpoint: str) -> Tuple[bool, Optional[str]]:
        """Perform HTTP health check against a given URL and endpoint.

        Returns:
            Tuple[bool, Optional[str]]: (is_healthy, error_message)
        """
        error_message: Optional[str] = None
        health_url = f"{url.rstrip('/')}/{endpoint.lstrip('/')}"
        is_healthy = False
        logger.debug(f"Checking HTTP health at {health_url}")
        try:
            response = requests.get(health_url, timeout=2)
            is_healthy = 200 <= response.status_code < 300
            if not is_healthy:
                 error_message = f"Health check failed (Status: {response.status_code}) at {health_url}"
        except requests.Timeout:
             error_message = f"Health check timed out at {health_url}"
             logger.debug(error_message)
        except requests.RequestException as e:
             error_message = f"Health check connection failed at {health_url}: {e}"
             logger.debug(error_message)
        except Exception as e:
             error_message = f"Unexpected error during health check at {health_url}: {e}"
             logger.error(error_message, exc_info=True)

        return is_healthy, error_message

    async def _check_slurm_status(self) -> Tuple[Optional[JobState], Optional[str], Optional[str]]:
        """Check Slurm status via scontrol/sacct.

        Returns:
            Tuple[Optional[JobState], Optional[str], Optional[str]]: (slurm_state, node, error_message)
        """
        logger.debug(f"Checking Slurm status for job {self.job_status.job_id}")
        job_id = self.job_status.job_id
        slurm_state: Optional[JobState] = None
        node: Optional[str] = self.job_status.node # Default to last known node
        error_message: Optional[str] = None

        try:
            scontrol_info, scontrol_stderr, scontrol_rc = await run_async_scontrol(job_id)
            if scontrol_rc == 0:
                node = scontrol_info.get('NodeList', node)
                status_str = scontrol_info.get('JobState', 'UNKNOWN')
                try: slurm_state = JobState(status_str)
                except ValueError: slurm_state = JobState.UNKNOWN
                if slurm_state.is_finished:
                     error_message = scontrol_info.get('Reason', f'Job {slurm_state.value}')
                return slurm_state, node, error_message
            elif "Invalid job id specified" in (scontrol_stderr or ""):
                logger.info(f"Job {job_id} not found by scontrol, querying sacct...")
                sacct_info, sacct_stderr, sacct_rc = await run_async_sacct(job_id)
                if sacct_rc == 0 and sacct_info:
                    status_str = sacct_info.get('State', 'UNKNOWN')
                    status_str_cleaned = status_str.split()[0]
                    try: slurm_state = JobState(status_str_cleaned)
                    except ValueError: slurm_state = JobState.UNKNOWN
                    logger.info(f"Job {job_id} final state from sacct: {slurm_state.value}")
                    if slurm_state == JobState.FAILED:
                        exit_code = sacct_info.get('ExitCode')
                        error_message = f"Job failed according to Slurm accounting (ExitCode: {exit_code or 'N/A'})"
                    elif not slurm_state.is_finished:
                         slurm_state = JobState.UNKNOWN
                         error_message = "Inconsistent state between scontrol and sacct"
                else:
                    slurm_state = JobState.UNKNOWN
                    error_message = f"Could not retrieve final job status from sacct: {sacct_stderr or 'Query failed'}"
                    logger.warning(f"sacct query failed or did not find job {job_id}. RC={sacct_rc}, Stderr={sacct_stderr}")
                return slurm_state, node, error_message
            else:
                slurm_state = JobState.UNKNOWN
                error_message = f"scontrol query failed: {scontrol_stderr or 'Unknown error'}"
                logger.error(f"scontrol failed for job {job_id}: {scontrol_stderr}")
                return slurm_state, node, error_message
        except Exception as e:
            logger.error(f"Unexpected error checking Slurm status for job {job_id}: {e}", exc_info=True)
            return JobState.UNKNOWN, node, f"Unexpected error checking Slurm status: {e}"

    async def get_status(self) -> JobStatus:
        """
        Get the status of the vLLM job, check relevant sources (Slurm/HTTP),
        and update the internal job_status object.
        """
        updated_status = self.job_status.copy(deep=True)
        updated_status.updated_at = datetime.utcnow()
        updated_status.error_message = None # Clear previous high-level error

        final_status = JobState.UNKNOWN
        check_error: Optional[str] = None
        node = updated_status.node # Keep current node unless updated by Slurm check

        if updated_status.is_static:
            # Static job: Only check HTTP status
            server_url = updated_status.server_url
            health_endpoint = "/health" # Default for vLLM
            # Find corresponding static config for specific health endpoint if configured
            static_config = next((m for m in settings.static_models if m.id == self.job_status.job_id), None)
            if static_config and hasattr(static_config, 'health_endpoint') and static_config.health_endpoint:
                health_endpoint = static_config.health_endpoint

            if server_url:
                 is_healthy, check_error = await self._check_http_status(server_url, health_endpoint)
                 final_status = JobState.RUNNING if is_healthy else JobState.FAILED
            else:
                 check_error = "Server URL not configured for static job"
                 final_status = JobState.UNKNOWN
        else:
            # Slurm job: Check Slurm first, then HTTP if running
            slurm_state, node_from_slurm, check_error = await self._check_slurm_status()
            node = node_from_slurm or node # Update node if Slurm provided it

            if slurm_state is None: slurm_state = JobState.UNKNOWN # Handle unexpected error case
            final_status = slurm_state # Start with Slurm's reported state

            # If Slurm reports RUNNING, perform secondary health check
            if slurm_state == JobState.RUNNING:
                server_url = updated_status.server_url # Use URL potentially updated by RM
                if server_url:
                    health_endpoint = "/health" # vLLM default
                    is_healthy, health_error = await self._check_http_status(server_url, health_endpoint)
                    if not is_healthy:
                        final_status = JobState.STARTING # Downgrade to STARTING if health check fails
                        # Combine error messages
                        check_error = f"{health_error}. {check_error}" if check_error else health_error
                        logger.debug(f"Job {self.job_status.job_id} Slurm state RUNNING, but health check failed: {health_error}")
                else:
                    # If RUNNING but no URL, consider it STARTING
                    final_status = JobState.STARTING
                    no_url_msg = "Job RUNNING in Slurm but server URL not available."
                    check_error = f"{no_url_msg} {check_error}" if check_error else no_url_msg
                    logger.warning(f"Job {self.job_status.job_id}: {no_url_msg}")

        # Update the JobStatus object
        updated_status.status = final_status
        updated_status.node = node # Use potentially updated node
        updated_status.error_message = check_error # Store combined error from checks

        # Persist the updated status internally
        self.job_status = updated_status
        return updated_status

    async def terminate(self) -> bool:
        """
        Terminate the vLLM job.
        Raises NotImplementedError if called on a static job.
        """
        if self.job_status.is_static:
            logger.error(f"Attempted to terminate static vLLM job {self.job_status.job_id}")
            raise NotImplementedError("Cannot terminate a statically configured vLLM service.")

        if self.job_status.status.is_finished:
             logger.info(f"Job {self.job_status.job_id} is already in terminal state {self.job_status.status}. No termination needed.")
             return True
        logger.info(f"Attempting to terminate SLURM job {self.job_status.job_id}")
        try:
            stdout, stderr, return_code = await run_async_scancel(self.job_status.job_id)
            if return_code == 0:
                logger.info(f"Successfully executed scancel for job {self.job_status.job_id}")
                return True
            else:
                logger.error(f"scancel command failed for job {self.job_status.job_id}. Return Code: {return_code}. Stderr: {stderr}")
                return False
        except Exception as e:
            logger.error(f"Exception during scancel for job {self.job_status.job_id}: {e}", exc_info=True)
            return False

    async def get_schema_url(self) -> Optional[str]:
        """
        Get the OpenAPI schema URL for the running vLLM server.
        """
        if self.job_status.status.is_active and self.job_status.server_url:
            schema_url = f"{self.job_status.server_url.rstrip('/')}/openapi.json"
            logger.debug(f"Potential schema URL for job {self.job_status.job_id}: {schema_url}")
            return schema_url
        else:
            logger.debug(f"Cannot get schema URL for job {self.job_status.job_id} (Status: {self.job_status.status}, URL: {self.job_status.server_url})")
            return None