from typing import Dict, Optional, List, Set, Type, Tuple
import subprocess
import json
import socket
import logging
import requests
from pathlib import Path
from fastapi import HTTPException
from core.models import SlurmConfig, VLLMConfig, JobStatus, GenericLaunchConfig, JobState
from core.resource_manager import resource_manager
import asyncio
import filelock
from pydantic import BaseModel, Field
import aiofiles
from datetime import datetime

# service runs on an Ubuntu HPC computer
# service is run alongside an NGINX reverse proxy which forwards vLLM api requests to associated
# vllm servers based on model name. RosieLLM api requests are routed to the RosieLLM api instead

# Configure logging
logger = logging.getLogger(__name__)

# Constants
ROSIE_WEB_DOMAIN = "https://dh-ood.hpc.msoe.edu"
ROSIE_BASE_URL_TEMPLATE = "/node/{node_url}.hpc.msoe.edu/{port}"
NGINX_CONF_PATH = "/etc/nginx/conf.d/vllm.conf" #TODO ensure path is correct
SBATCH_DIR = Path("/data/ai_club/RosieLLM/sbatch") #TODO ensure path is correct, update ROSIE config to Osire
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
    if slurm_config.additional_args:
        for key, value in slurm_config.additional_args.items():
            script_content += f"#SBATCH --{key}={value}\n"

    script_content += """
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
    --log-level {slurm_config.output_config.log_level}"""

    # Add any additional vLLM arguments if specified
    if vllm_config.additional_args:
        for key, value in vllm_config.additional_args.items():
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
        
        # Submit the job using sbatch
        result = subprocess.run(['sbatch', script_path], 
                            capture_output=True, 
                            text=True,
                            check=True)
        
        # Extract job ID from sbatch output (format: "Submitted batch job 123456")
        try:
            job_id = result.stdout.strip().split()[-1]
            int(job_id)  # Validate that job_id is a number
        except (IndexError, ValueError):
            raise HTTPException(status_code=500, 
                              detail=f"Invalid job ID from sbatch: {result.stdout}")
        
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

async def terminate_job(job_status: JobStatus) -> None:
    """Terminate a SLURM job."""
    logger.info(f"Terminating job {job_status.job_id}")
    try:
        # Use asyncio.gather to perform operations atomically
        async with resource_manager._job_lock:  # Prevent job status changes during termination
            # Cancel the SLURM job
            process = await asyncio.create_subprocess_exec(
                'scancel', 
                job_status.job_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, 'scancel', stdout, stderr)
            
            # Update NGINX configuration to remove the server
            if job_status.node and job_status.port:
                upstream = f"{job_status.node}:{job_status.port}"
                logger.debug(f"Removing NGINX upstream {upstream}")
                await _remove_nginx_upstream(upstream)

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to terminate job: {e.stderr}")
        raise HTTPException(status_code=500, 
                          detail=f"Failed to terminate job: {e.stderr}")

async def _update_nginx_config(nginx_upstream: str, remove: bool = False) -> None:
    """Update NGINX configuration by adding or removing an upstream server."""
    logger.info(f"{'Removing' if remove else 'Adding'} NGINX upstream {nginx_upstream}")
    try:
        # Use asyncio lock instead of filelock for better async support
        nginx_lock = asyncio.Lock()
        async with nginx_lock:
            if remove:
                async with aiofiles.open(NGINX_CONF_PATH, 'r') as f:
                    config = await f.readlines()
                config = [line for line in config if nginx_upstream not in line]
                async with aiofiles.open(NGINX_CONF_PATH, 'w') as f:
                    await f.writelines(config)
            else:
                async with aiofiles.open(NGINX_CONF_PATH, 'r') as f:
                    config = await f.read()
                if nginx_upstream not in config:
                    upstream_block = config.find('upstream vllm {')
                    if upstream_block != -1:
                        insert_pos = config.find('}', upstream_block)
                        config = config[:insert_pos] + f"    server {nginx_upstream};\n" + config[insert_pos:]
                        async with aiofiles.open(NGINX_CONF_PATH, 'w') as f:
                            await f.write(config)

            # Reload NGINX
            await asyncio.create_subprocess_exec('nginx', '-s', 'reload')

    except Exception as e:
        action = "remove from" if remove else "update"
        error_msg = f"Failed to {action} NGINX config: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

async def update_nginx_config(nginx_upstream: str) -> None:
    """Update NGINX configuration with new upstream server."""
    await _update_nginx_config(nginx_upstream, remove=False)

async def _remove_nginx_upstream(nginx_upstream: str) -> None:
    """Remove an upstream server from NGINX configuration."""
    await _update_nginx_config(nginx_upstream, remove=True)

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
        vllm_config_dict["additional_args"] = vllm_extra_args
    if slurm_extra_args:
        slurm_config_dict["additional_args"] = slurm_extra_args
    
    # Create config objects
    vllm_config = VLLMConfig(**vllm_config_dict)
    slurm_config = SlurmConfig(**slurm_config_dict)
    
    logger.debug(f"Split configs - vLLM: {vllm_config}, SLURM: {slurm_config}")
    return vllm_config, slurm_config

def check_vllm_server(node: str, port: int) -> bool:
    """Check if vLLM server is healthy"""
    url = f"{ROSIE_WEB_DOMAIN}{ROSIE_BASE_URL_TEMPLATE.format(node_url=node, port=port)}/health"
    logger.debug(f"Checking vLLM server health at {url}")
    try:
        response = requests.get(url, timeout=2)
        is_healthy = response.status_code == 200
        logger.debug(f"vLLM server health check {'succeeded' if is_healthy else 'failed'}")
        return is_healthy
    except requests.RequestException as e:
        logger.debug(f"vLLM server health check failed: {str(e)}")
        return False

async def get_job_info(job_id: str, port: int) -> Dict:
    """Get detailed information about a SLURM job."""
    logger.debug(f"Getting info for job {job_id}")
    try:
        # Get SLURM job info with timeout
        result = await asyncio.wait_for(
            asyncio.create_subprocess_shell(
                f'scontrol show job {job_id}',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            ),
            timeout=5.0
        )
        stdout, stderr = await result.communicate()
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, 'scontrol', stdout, stderr)
        
        info = dict(item.split('=', 1) for item in stdout.decode().split() if '=' in item)
        node = info.get('NodeList', None)
        status = JobState.UNKNOWN
        
        # Determine job status
        if info.get('JobState') == "RUNNING":
            status = (JobState.RUNNING if node and check_vllm_server(node, port) 
                     else JobState.STARTING)
        else:
            try:
                status = JobState(info.get('JobState', 'UNKNOWN'))
            except ValueError:
                logger.warning(f"Unknown SLURM status: {info.get('JobState')}")
        
        return {
            'status': status,
            'node': node,
            'error_message': info.get('Reason') if status == JobState.FAILED else None,
            'updated_at': datetime.utcnow()
        }
        
    except asyncio.TimeoutError:
        logger.error(f"Timeout getting info for job {job_id}")
        raise HTTPException(status_code=500, detail="SLURM command timed out")
    except Exception as e:
        logger.error(f"Failed to get job info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))