from typing import Dict, Optional, List, Type, Tuple, Any
import logging
from fastapi import HTTPException, FastAPI
from core.models import SlurmConfig, VLLMConfig, JobStatus, JobState, LaunchRequest
from core.resource_manager import resource_manager
import asyncio
import httpx
import re
from .engines import factory as engine_factory
from core.settings import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Constants
# SBATCH_DIR = Path("/data/ai_club/RosieLLM/out") # TODO: This seems engine-specific, should not be here
# SBATCH_DIR.mkdir(parents=True, exist_ok=True)

# Track if API docs have been refreshed
_docs_refreshed = False

async def launch_job(request: LaunchRequest) -> JobStatus:
    """Orchestrates the launch of a new inference job."""
    model_name = request.model_name
    engine_type = request.engine_type.lower()
    logger.info(f"Service layer: Received launch request for model '{model_name}' using engine '{engine_type}'")

    # 1. Check for existing active job for this model name
    try:
        existing_job = await resource_manager.get_job(model_name)
        # If get_job succeeds, a job exists. Check if it's active.
        if existing_job and existing_job.status.is_active:
            # Policy: Disallow launching if an active job exists for the same model name
            logger.warning(f"Conflict: Model '{model_name}' already has an active job {existing_job.job_id}.")
            raise HTTPException(status_code=409, detail=f"Model '{model_name}' already has an active job ({existing_job.job_id}). Terminate it first or wait.")
        elif existing_job:
            logger.info(f"An inactive job exists for '{model_name}' ({existing_job.job_id}, status: {existing_job.status}). Proceeding with new launch.")
            # Optional: Could remove the old job entry here if desired
            # await resource_manager.remove_job(model_name)
    except HTTPException as e:
        if e.status_code == 404:
            logger.info(f"No existing job found for model '{model_name}'. Proceeding with launch.")
            # This is the expected case for a new launch
        else:
            # Re-raise other HTTP exceptions from get_job (e.g., internal errors)
            raise e

    # 2. Orchestrate launch via factory and resource manager primitives
    try:
        # 2a. Validate args using factory
        # config_objs contains validated Pydantic models specific to the engine
        config_objs = engine_factory.validate_launch_args(engine_type, request.engine_args)
        logger.debug(f"Launch args validated successfully for {engine_type}.")

        # 2b. Acquire port from resource manager
        port = await resource_manager.acquire_port()
        logger.debug(f"Acquired port {port} for job.")

        # 2c. Generate script using factory (if applicable)
        script_content = ""
        job_name = f"OsireLLM_{model_name.replace('/','_')}_{port}" # Default
        try:
            script_content, generated_job_name = engine_factory.generate_script(engine_type, model_name, port, *config_objs)
            job_name = generated_job_name # Use engine-generated name if provided
            logger.debug(f"Script generated for job '{job_name}'.")
        except NotImplementedError:
            logger.info(f"Engine type '{engine_type}' does not require script generation.")

        # 2d. Submit launch command using factory
        job_id, error_message = await engine_factory.submit_launch(engine_type, script_content, job_name)

        if error_message or not job_id:
            logger.error(f"Engine launch submission failed: {error_message}")
            # TODO: Release acquired port if submission fails?
            raise HTTPException(status_code=500, detail=f"Failed to submit job: {error_message or 'Unknown submission error'}")

        # 2e. Construct initial JobStatus
        # Extract necessary details from validated config_objs (engine-specific)
        # TODO: Standardize how details like num_gpus, partition are retrieved post-validation
        num_gpus = 1 # Default fallback
        partition = "unknown" # Default fallback
        if engine_type == "vllm" and len(config_objs) == 2:
            # Assuming config_objs = (VLLMConfig, SlurmConfig) for vLLM
            # This coupling is slightly awkward, maybe factory should return a dict?
            slurm_config = config_objs[1]
            num_gpus = getattr(slurm_config, 'gpus', 1)
            partition = getattr(slurm_config, 'partition', 'unknown')
        # Add logic for other engine types if needed

        initial_job_status = JobStatus(
            job_id=job_id,
            model_name=model_name,
            engine_type=engine_type,
            num_gpus=num_gpus,
            partition=partition,
            status=JobState.PENDING,
            port=port,
            is_static=False,
            # owner, node, server_url, created_at, updated_at are set by model/manager
        )
        logger.info(f"Job submitted successfully via engine: {initial_job_status}")

        # 2f. Add job to state manager and start updates
        await resource_manager.add_job(model_name, initial_job_status)

        # 2g. Return the initial status
        return initial_job_status

    # --- Error Handling --- #
    except ValueError as e: # Catch validation errors from factory
        logger.warning(f"Launch request validation failed: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid launch arguments: {e}")
    except NotImplementedError as e:
        logger.error(f"Launch failed: Engine type '{engine_type}' does not support required operation: {e}")
        raise HTTPException(status_code=501, detail=f"Engine '{engine_type}' does not support launch: {e}")
    except HTTPException as he: # Re-raise HTTP exceptions (e.g., port acquisition failure, job conflict)
         raise he
    except Exception as e:
        # Catch-all for unexpected errors during orchestration
        logger.error(f"Unexpected error during job launch orchestration for model '{model_name}': {e}", exc_info=True)
        # TODO: Release acquired port if possible?
        raise HTTPException(status_code=500, detail=f"Unexpected internal error launching job: {e}")

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
