from typing import List, Optional, Dict
import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
import asyncio

from core.models import SlurmConfig, VLLMConfig, JobStatus, GenericLaunchConfig, JobState
from core.resource_manager import JobStateManager, get_resource_manager
from core.osire_llm_service import launch_server, split_config, refresh_api_docs_with_model, has_docs_been_refreshed

# Configure logging
logger = logging.getLogger(__name__)

# Create router for vLLM management
router = APIRouter(tags=["OsireLLM"], prefix="")

@router.post("/launch", response_model=JobStatus)
async def launch_vllm_server(
    request: Request,
    config: GenericLaunchConfig,
    vllm_extra_args: Optional[Dict] = None,
    slurm_extra_args: Optional[Dict] = None,
    job_manager: JobStateManager = Depends(get_resource_manager)
) -> JSONResponse:
    """Launch a new vLLM server instance"""
    logger.info(f"Launching vLLM server for model {config.model_name}")
    try:
        # Split generic config into specific configs
        vllm_config, slurm_config = split_config(
            config,
            vllm_extra_args,
            slurm_extra_args
        )
        # Launch server
        job_status = await launch_server(vllm_config, slurm_config)
        await job_manager.add_job(vllm_config.model_name, job_status)
        logger.info(f"Server launched successfully: {job_status}")

        # Start docs refresh task if not already done
        if not has_docs_been_refreshed():
            logger.info("First model launch detected, will refresh API docs when model is running")
            asyncio.create_task(
                refresh_api_docs_with_model(
                    app=request.app,
                    model_name=vllm_config.model_name
                )
            )

        return JSONResponse(status_code=201, content=job_status.model_dump())
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Failed to launch server: {str(e)}")
        if "resource" in str(e).lower():
            raise HTTPException(status_code=503, detail=f"Resource unavailable: {str(e)}")
        elif "permission" in str(e).lower():
            raise HTTPException(status_code=403, detail=f"Permission denied: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail=str(e))

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

@router.delete("/terminate/")
async def terminate_vllm_job(
    model_name: str,
    job_manager: JobStateManager = Depends(get_resource_manager)
) -> JSONResponse:
    """Terminate a specific vLLM job"""
    logger.info(f"Terminating job for model {model_name}")
    try:
        job_status = await job_manager.get_job(model_name)
        await job_manager.terminate_job(model_name)

        logger.info(f"Successfully terminated job for model {model_name}")
        return JSONResponse(
            status_code=200,
            content={"message": f"Job for model {model_name} terminated successfully",
                    "job_status": job_status.model_dump()
            }
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Failed to terminate job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))