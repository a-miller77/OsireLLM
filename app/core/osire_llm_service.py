from typing import Dict, Optional, List, Type, Tuple, Any
import subprocess
import logging
from pathlib import Path
from fastapi import HTTPException, FastAPI
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
import asyncio
import httpx
import re

# Configure logging
logger = logging.getLogger(__name__)

# Constants
SBATCH_DIR = Path("/data/ai_club/RosieLLM/out") #TODO ensure path is correct, update ROSIE config to Osire
SBATCH_DIR.mkdir(parents=True, exist_ok=True) #TODO ensure correct

# Track if API docs have been refreshed
_docs_refreshed = False

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

async def fetch_openapi_schema(endpoint: str) -> Dict[str, Any]:
    """Fetch the OpenAPI schema from a model server"""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{endpoint}/openapi.json")
            if response.status_code != 200:
                logger.error(f"Failed to fetch OpenAPI schema: {response.status_code}")
                return {}
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching OpenAPI schema: {str(e)}")
        return {}

def update_app_schema_with_model_api(
    app: FastAPI,
    schema: Dict[str, Any],
    tag: str = "default"
):
    """Merge an external OpenAPI schema into the FastAPI app's schema"""
    if not schema:
        logger.warning("Empty schema provided, skipping OpenAPI update")
        return

    logger.info(f"Starting OpenAPI schema update with tag '{tag}'")

    # Clear the cached schema to force regeneration with our modifications
    app.openapi_schema = None

    # Store the original openapi method
    original_openapi = app.openapi

    def custom_openapi():
        # Check if we already have a cached schema
        if app.openapi_schema:
            return app.openapi_schema

        # Generate the base schema using the original method
        current_schema = original_openapi()

        # Explicitly save the servers configuration
        servers_config = current_schema.get("servers", None)
        logger.info(f"Preserved servers configuration: {servers_config}")

        # Only include PUT and POST operations
        supported_operations = ["post", "put"]

        # Track which components are needed by the endpoints we're adding
        needed_components = set()

        # Helper function to collect references
        def collect_references(obj, refs_set):
            """Recursively collect all $ref values from an object"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "$ref" and isinstance(value, str) and value.startswith("#/components/"):
                        refs_set.add(value)
                    elif isinstance(value, (dict, list)):
                        collect_references(value, refs_set)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        collect_references(item, refs_set)

        # First pass: identify paths to add and collect needed components
        paths_to_add = {}
        if "paths" in schema:
            # Regex pattern to match /v1, /v2, etc. at the start of the path
            version_pattern = re.compile(r"^/v\d+")

            for path, path_item in schema["paths"].items():
                # Only include paths that start with /v followed by a number
                if version_pattern.match(path):
                    # Only add if the path doesn't already exist
                    if path not in current_schema.get("paths", {}):
                        # Create a modified path item with only supported operations
                        modified_path_item = {}

                        # Only copy supported operations (POST and PUT)
                        for operation in supported_operations:
                            if operation in path_item:
                                # Copy the operation and set our tag
                                modified_path_item[operation] = path_item[operation].copy()
                                modified_path_item[operation]["tags"] = [tag]

                                # Collect schema references from this operation
                                collect_references(path_item[operation], needed_components)

                        # Only add the path if it has at least one supported operation
                        if modified_path_item:
                            paths_to_add[path] = modified_path_item

        logger.info(f"Found {len(paths_to_add)} paths to add from model schema")
        logger.debug(f"Paths to add: {list(paths_to_add.keys())}")
        logger.info(f"Found {len(needed_components)} component references to process")

        # Add only the needed components
        components_added = 0
        if "components" in schema and needed_components:
            if "components" not in current_schema:
                current_schema["components"] = {}

            # Process components in multiple passes to handle nested references
            processed = set()
            while needed_components - processed:
                new_refs = set()

                for ref in needed_components - processed:
                    # Parse the reference to get component type and name
                    # Format is #/components/{type}/{name}
                    parts = ref.split('/')
                    if len(parts) >= 4 and parts[1] == "components":
                        component_type = parts[2]
                        component_name = parts[3]

                        # Add this component if it exists and we don't already have it
                        if (component_type in schema.get("components", {}) and
                            component_name in schema["components"][component_type]):

                            # Ensure the component type exists in our schema
                            if component_type not in current_schema["components"]:
                                current_schema["components"][component_type] = {}

                            # Only add if we don't already have this component
                            if component_name not in current_schema["components"][component_type]:
                                component_def = schema["components"][component_type][component_name]
                                current_schema["components"][component_type][component_name] = component_def
                                components_added += 1
                                logger.debug(f"Added component: {component_type}/{component_name}")

                                # Look for more references in this component
                                collect_references(component_def, new_refs)

                    # Mark this reference as processed
                    processed.add(ref)

                # Add any new references we found
                if new_refs:
                    logger.debug(f"Found {len(new_refs)} new component references")
                    needed_components.update(new_refs)

        logger.info(f"Added {components_added} components to schema")

        # Add the paths
        if paths_to_add:
            if "paths" not in current_schema:
                current_schema["paths"] = {}
            for path, path_item in paths_to_add.items():
                current_schema["paths"][path] = path_item
                logger.debug(f"Added path: {path}")

        # Explicitly restore the servers configuration
        if servers_config:
            if "servers" not in current_schema or current_schema["servers"] != servers_config:
                logger.info(f"Restoring servers configuration: {servers_config}")
                current_schema["servers"] = servers_config
            else:
                logger.info("Servers configuration is already set, skipping restore")
        else:
            logger.warning("No servers configuration to restore")

        # Cache the modified schema
        app.openapi_schema = current_schema
        logger.info("Successfully updated OpenAPI schema")
        return app.openapi_schema

    # Replace the openapi method with our custom one
    app.openapi = custom_openapi

async def refresh_api_docs_with_model(app: FastAPI, model_name: str, timeout_seconds: int = 180):
    """
    Wait for model to be running and then refresh API docs.

    Args:
        app: The FastAPI app instance
        model_name: The name of the model to use for refreshing docs
        timeout_seconds: Maximum time to wait for model to become available
    """
    global _docs_refreshed

    logger.info(f"Starting API docs refresh task for model {model_name}")
    start_time = asyncio.get_event_loop().time()

    while True:
        # Check timeout
        if asyncio.get_event_loop().time() - start_time > timeout_seconds:
            logger.warning(f"Timed out waiting for model {model_name} to become available for docs refresh")
            return

        try:
            # Check if model is running
            job_status = await resource_manager.get_job(model_name)
            if job_status.status == JobState.RUNNING:
                logger.info(f"Model {model_name} is running, refreshing API docs")

                # Fetch schema and update docs
                model_schema = await fetch_openapi_schema(job_status.server_url)
                update_app_schema_with_model_api(app, model_schema, tag="LLM")

                _docs_refreshed = True
                logger.info(f"Successfully refreshed API docs using model {model_name}")
                return

            # Wait before checking again
            await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Error while trying to refresh docs with model {model_name}: {str(e)}")
            # Wait before retrying
            await asyncio.sleep(10)

# Function to check if docs have been refreshed
def has_docs_been_refreshed() -> bool:
    """Return whether docs have been refreshed at least once"""
    global _docs_refreshed
    return _docs_refreshed

# Function to initiate manual docs refresh
async def manually_refresh_docs_with_model(app: FastAPI, model_name: str) -> Dict[str, Any]:
    """
    Manually refresh API docs using a specified model

    Args:
        app: The FastAPI app instance
        model_name: The name of the model to use

    Returns:
        Dict with status information
    """
    global _docs_refreshed

    try:
        # Check if model is running
        job_status = await resource_manager.get_job(model_name)
        if job_status.status != JobState.RUNNING:
            return {
                "success": False,
                "message": f"Model {model_name} is not running"
            }

        # Fetch schema and update docs
        model_schema = await fetch_openapi_schema(job_status.server_url)
        update_app_schema_with_model_api(app, model_schema, tag="LLM")

        _docs_refreshed = True
        return {
            "success": True,
            "message": f"Successfully refreshed API docs using model {model_name}"
        }
    except Exception as e:
        logger.error(f"Error manually refreshing docs: {str(e)}")
        return {
            "success": False,
            "message": f"Error refreshing docs: {str(e)}"
        }
