from typing import List, Optional, Dict
import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
import asyncio

from core.models import JobStatus, JobState, LaunchRequest
from core.resource_manager import get_resource_manager, JobStateManager
from core.osire_llm_service import launch_job, refresh_api_docs_with_model, has_docs_been_refreshed

# Configure logging
logger = logging.getLogger(__name__)

# Create router for vLLM management
router = APIRouter(tags=["OsireLLM"], prefix="")

@router.post("/launch", response_model=JobStatus, status_code=201)
async def launch_model_server(
    request: Request,
    launch_request: LaunchRequest
) -> JobStatus:
    """Launch a new inference server based on the provided configuration."""
    model_name = launch_request.model_name
    engine_type = launch_request.engine_type
    logger.info(f"Route: Received request to launch model '{model_name}' using engine '{engine_type}'")
    try:
        job_status = await launch_job(launch_request)
        logger.info(f"Route: Service layer reported successful launch: {job_status}")

        if not has_docs_been_refreshed():
            logger.info(f"First model launch detected ({model_name}), scheduling API docs refresh task.")
            asyncio.create_task(
                refresh_api_docs_with_model(
                    app=request.app,
                    model_name=model_name
                )
            )

        return job_status
    except HTTPException as he:
        logger.warning(f"Route: Launch failed for '{model_name}'. Reason: {he.detail} (Status: {he.status_code})")
        raise he
    except Exception as e:
        logger.error(f"Route: Unexpected error launching '{model_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error launching job: {e}")

@router.get("/status/")
async def get_job_status(
    job_manager: JobStateManager = Depends(get_resource_manager)
) -> JSONResponse:
    """Get status of all jobs"""
    logger.debug("Getting status for all jobs")
    try:
        jobs = await job_manager.get_all_jobs()
        return JSONResponse(
            status_code=200,
            content={"jobs": [job.model_dump() for job in jobs.values()]}
        )
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models(
    job_manager: JobStateManager = Depends(get_resource_manager)
) -> JSONResponse:
    """List all currently running models"""
    logger.debug("Listing running models")
    try:
        jobs = await job_manager.get_all_jobs()
        # Only include models that are in a RUNNING state
        models = [
            model_name for model_name, job in jobs.items()
            if job.status == JobState.RUNNING
        ]

        logger.debug(f"Found running models: {models}")
        return JSONResponse(status_code=200, content=models)
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/terminate/{model_name}")
async def terminate_vllm_job(
    model_name: str,
    job_manager: JobStateManager = Depends(get_resource_manager)
) -> JSONResponse:
    """Terminate a specific inference job by model name."""
    logger.info(f"Route: Received request to terminate job for model '{model_name}'")
    try:
        await job_manager.terminate_job(model_name)
        logger.info(f"Route: Termination command issued successfully for model '{model_name}'")
        return JSONResponse(
            status_code=200,
            content={ "message": f"Termination request accepted for model '{model_name}'. Status will update."}
        )
    except HTTPException as he:
        logger.warning(f"Route: Termination failed for '{model_name}'. Reason: {he.detail} (Status: {he.status_code})")
        raise he
    except Exception as e:
        logger.error(f"Route: Unexpected error terminating '{model_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error terminating job: {e}")