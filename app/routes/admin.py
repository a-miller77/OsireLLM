from typing import Dict, Any
import logging
from fastapi import APIRouter, HTTPException, Depends, Request, Query
from fastapi.responses import JSONResponse
import asyncio

from core.resource_manager import JobStateManager, get_resource_manager
from core.models import JobState
from core.osire_llm_service import manually_refresh_docs_with_model

# Configure logging
logger = logging.getLogger(__name__)

# Create router for admin functions
router = APIRouter(tags=["admin"], prefix="")

@router.get("/health", include_in_schema=True)
async def health_check() -> JSONResponse:
    """
    Basic health check endpoint for the API
    """
    return JSONResponse(status_code=200, content={"status": "healthy"})

@router.post("/refresh-docs", include_in_schema=False)
async def manually_refresh_docs(
    request: Request,
    model: str = None,
    job_manager: JobStateManager = Depends(get_resource_manager)
):
    """
    Manually refresh API docs using a specified or any running model
    """
    try:
        # If no model specified, find any running model
        if not model:
            jobs = await job_manager.get_all_jobs()
            running_models = [name for name, job in jobs.items()
                             if job.status == JobState.RUNNING]

            if not running_models:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No running models available to refresh docs"}
                )

            model = running_models[0]

        # Use centralized function from osire_llm_service
        result = await manually_refresh_docs_with_model(request.app, model)

        if result["success"]:
            return JSONResponse(
                status_code=202,
                content={"message": result["message"]}
            )
        else:
            return JSONResponse(
                status_code=400,
                content={"error": result["message"]}
            )
    except Exception as e:
        logger.error(f"Failed to initiate docs refresh: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", include_in_schema=True)
async def get_stats(
    job_manager: JobStateManager = Depends(get_resource_manager)
) -> JSONResponse:
    """
    Get statistics about the API usage and resources
    """
    try:
        jobs = await job_manager.get_all_jobs()

        # Calculate basic stats
        stats = {
            "total_jobs": len(jobs),
            "running_jobs": sum(1 for job in jobs.values() if job.status == JobState.RUNNING),
            "starting_jobs": sum(1 for job in jobs.values() if job.status == JobState.STARTING),
            "queued_jobs": sum(1 for job in jobs.values() if job.status == JobState.PENDING),
            "failed_jobs": sum(1 for job in jobs.values() if job.status.is_failed),
            "models": [name for name, job in jobs.items() if job.status == JobState.RUNNING]
        }

        return JSONResponse(status_code=200, content=stats)
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/log-level", include_in_schema=True)
async def set_log_level(level: str) -> JSONResponse:
    """
    Dynamically change the log level of the entire API

    Valid levels: debug, info, warning, error, critical
    """
    try:
        # Convert to lowercase to be case-insensitive
        level = level.lower()

        # Validate the log level
        valid_levels = ["debug", "info", "warning", "error", "critical"]
        if level not in valid_levels:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid log level. Valid levels are: {', '.join(valid_levels)}"}
            )

        # Get the numeric value for the log level
        numeric_level = getattr(logging, level)

        # Set the log level for the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)

        # Also update all existing handlers
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)

        # Also set the log level for our module logger
        logger.setLevel(numeric_level)

        return JSONResponse(
            status_code=200,
            content={"message": f"Log level set to {level}"}
        )
    except Exception as e:
        logger.error(f"Failed to change log level: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))