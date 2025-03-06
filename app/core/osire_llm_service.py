from typing import Dict, Optional, List, Type, Tuple
import subprocess
import logging
from pathlib import Path
from fastapi import HTTPException
from pydantic import BaseModel
from core.models import SlurmConfig, VLLMConfig, JobStatus, JobState, GenericLaunchConfig
from core.resource_manager import (
    resource_manager,
    get_job_info,
    check_vllm_server,
    ROSIE_WEB_DOMAIN,
    ROSIE_BASE_URL_TEMPLATE
)
from core.shell_commands import run_async_sbatch

# Configure logging
logger = logging.getLogger(__name__)

# Constants
SBATCH_DIR = Path("/data/ai_club/RosieLLM/out") #TODO ensure path is correct, update ROSIE config to Osire
SBATCH_DIR.mkdir(parents=True, exist_ok=True) #TODO ensure correct

def _generate_slurm_script(vllm_config: VLLMConfig,
                         slurm_config: SlurmConfig,
                         port: int) -> str:
    """Generate a SLURM batch script for running a vLLM server."""
    logger.debug(f"Generating SLURM script for model {vllm_config.model_name} on port {port}")

    script_content = f"""#!/bin/bash
#SBATCH --job-name={slurm_config.job_name}
#SBATCH --partition={slurm_config.partition}
#SBATCH --nodes={slurm_config.nodes}
#SBATCH --gpus={slurm_config.gpus}
#SBATCH --cpus-per-gpu={slurm_config.cpus_per_gpu}
#SBATCH --time={slurm_config.time_limit}
#SBATCH --output={slurm_config.output_config.stdout_file}
"""

    # Add any additional SLURM arguments if specified
    if slurm_config.slurm_extra_args:
        for key, value in slurm_config.slurm_extra_args.items():
            script_content += f"#SBATCH --{key}={value}\n"

    script_content += f"""
# Load required modules
module load singularity

# Set environment variables

# Launch vLLM server using singularity
singularity exec --nv -B /data:/data {slurm_config.container} python3 -m vllm.entrypoints.openai.api_server \\
    --model {vllm_config.model_name} \\
    --download-dir {vllm_config.download_dir} \\
    --host 0.0.0.0 \\
    --port {port} \\
    --root-path {ROSIE_BASE_URL_TEMPLATE.format(
        node_url="$SLURMD_NODENAME",
        port=port
    )} \\
    --max-num-batched-tokens {vllm_config.max_num_batched_tokens} \\
    --gpu-memory-utilization {vllm_config.gpu_memory_utilization} \\
    --dtype {vllm_config.dtype} \\
    --max-model-len {vllm_config.max_model_len} \\
    --tensor-parallel-size {slurm_config.gpus} \\
    --pipeline-parallel-size {slurm_config.nodes} \\
    --uvicorn-log-level {slurm_config.output_config.log_level}"""

    # Add any additional vLLM arguments if specified
    if vllm_config.vllm_extra_args:
        for key, value in vllm_config.vllm_extra_args.items():
            script_content += f" \\\n    --{key} {value}"

    # Save the script
    script_path = SBATCH_DIR / f"{slurm_config.job_name}_{port}.sh"
    logger.debug(f"Saving SLURM script to {script_path}")
    with open(script_path, 'w') as f:
        f.write(script_content)

    return str(script_path)

async def launch_server(vllm_config: VLLMConfig,
                       slurm_config: SlurmConfig) -> JobStatus:
    """Launch a vLLM server using SLURM."""
    logger.info(f"Launching server for model {vllm_config.model_name}")
    try:
        # Check if model is already running before acquiring resources
        if await resource_manager.is_model_running(vllm_config.model_name):
            raise HTTPException(
                status_code=409,
                detail=f"Model {vllm_config.model_name} is already running"
            )

        # Find available port and generate script
        port = await resource_manager.acquire_port()
        script_path = _generate_slurm_script(vllm_config, slurm_config, port)
        logger.info(f"Generated SLURM script: {script_path}")

        # Run sbatch using our async SSH command utility
        stdout, stderr, return_code = await run_async_sbatch(script_path)

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, 'sbatch', stdout, stderr)

        # Extract job ID from sbatch output (format: "Submitted batch job 123456")
        try:
            job_id = stdout.strip().split()[-1]
            int(job_id)  # Validate that job_id is a number
        except (IndexError, ValueError):
            raise HTTPException(status_code=500,
                              detail=f"Invalid job ID from sbatch: {stdout}")

        job_status = JobStatus(
            job_id=job_id,
            status=JobState.PENDING,
            model_name=vllm_config.model_name,
            num_gpus=slurm_config.gpus,
            partition=slurm_config.partition,
            node=None,  # Will be set once job is running
            port=port
        )
        logger.info(f"Server submitted successfully: {job_status}")
        return job_status

    except Exception as e:
        error_msg = str(e)
        if isinstance(e, subprocess.CalledProcessError):
            error_msg = e.stderr
        elif isinstance(e, HTTPException):
            error_msg = e.detail

        logger.error(f"Failed to launch job: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Failed to launch job: {error_msg}")

def split_config(
    generic_config: GenericLaunchConfig,
    vllm_extra_args: Optional[Dict] = None,
    slurm_extra_args: Optional[Dict] = None
) -> Tuple[VLLMConfig, SlurmConfig]:
    """Split generic config into vLLM and SLURM specific configs"""
    logger.debug("Splitting generic config into specific configs")

    def get_model_fields(model_class: Type[BaseModel]) -> set:
        """Get non-private field names from a model class"""
        return {
            name for name, field in model_class.model_fields.items()
            if not name.startswith('_')
        }

    # Get field names for each config type
    vllm_fields = get_model_fields(VLLMConfig)
    slurm_fields = get_model_fields(SlurmConfig)

    # Extract fields for each config from generic config
    vllm_config_dict = {
        field: getattr(generic_config, field)
        for field in vllm_fields
        if hasattr(generic_config, field)
    }

    slurm_config_dict = {
        field: getattr(generic_config, field)
        for field in slurm_fields
        if hasattr(generic_config, field)
    }

    # Add extra arguments if provided
    if vllm_extra_args:
        vllm_config_dict["vllm_extra_args"] = vllm_extra_args
    if slurm_extra_args:
        slurm_config_dict["slurm_extra_args"] = slurm_extra_args

    # Create config objects
    vllm_config = VLLMConfig(**vllm_config_dict)
    slurm_config = SlurmConfig(**slurm_config_dict)

    logger.debug(f"Split configs - vLLM: {vllm_config}, SLURM: {slurm_config}")
    return vllm_config, slurm_config
