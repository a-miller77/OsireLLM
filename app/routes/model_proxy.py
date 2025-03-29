from typing import Optional, Dict, Any
import logging
import re
import json
from fastapi import APIRouter, HTTPException, Depends, Request, Response, Query, Path
from fastapi.responses import StreamingResponse
import httpx

from core.resource_manager import JobStateManager, get_resource_manager
from core.models import JobState

# Configure logging
logger = logging.getLogger(__name__)

# Create router for model proxying with version prefix
router = APIRouter(tags=["LLM"])

async def get_model_endpoint(
    model: str,
    job_manager: JobStateManager = Depends(get_resource_manager)
) -> Dict[str, Any]:
    """Get the endpoint URL for a running model server"""
    job_status = await job_manager.get_job(model)

    if job_status.status != JobState.RUNNING:
        raise HTTPException(
            status_code=400,
            detail=f"Model server for {model} is not in RUNNING state"
        )

    if not job_status.server_url:
        raise HTTPException(
            status_code=404,
            detail=f"No endpoint available for model server {model}"
        )

    return {
        "url": job_status.server_url,
        "type": "vllm"  # Default to vLLM
    }

@router.api_route("/v{version}/{path:path}", methods=["POST", "PUT"], include_in_schema=False)
async def proxy_to_model(
    request: Request,
    version: int = Path(..., description="API version"),
    path: str = Path(..., description="API path"),
    model: Optional[str] = Query(None, description="Model to use for this request"),
    job_manager: JobStateManager = Depends(get_resource_manager)
):
    """Proxy POST/PUT requests to model server with OpenAI-compatible routing"""
    try:
        # Get request body for model extraction if needed
        body_bytes = await request.body()

        if not model:
            # Try to extract model from request body (OpenAI-style)
            try:
                body_str = body_bytes.decode('utf-8')
                logger.debug(f"Request body: {body_str}")
                body_json = json.loads(body_bytes)
                model = body_json.get("model")
                logger.debug(f"Extracted model from request body: {model}")
            except Exception as e:
                logger.debug(f"Could not extract model from request body: {str(e)}")

        # Require a model parameter (no defaults)
        if not model:
            logger.warning("No model parameter found in query or request body")
            raise HTTPException(
                status_code=400,
                detail="Model parameter is required (either in query or request body)"
            )

        # Get the model endpoint
        endpoint_info = await get_model_endpoint(model, job_manager)
        endpoint_url = endpoint_info["url"]

        # Construct target URL (include the version in the path)
        full_path = f"v{version}/{path}"
        target_url = f"{endpoint_url}/{full_path}"
        logger.info(f"Proxying request to: {target_url} for model {model}")

        # Get request headers but exclude host header
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

        # Get query parameters but exclude 'model' since we already processed it
        params = {k: v for k, v in request.query_params.items() if k != "model"}

        # Create httpx client
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=600)) as client:
            # Check if request involves streaming
            is_streaming = "stream" in path or params.get("stream") == "true"

            if is_streaming:
                # For streaming responses
                async def stream_response():
                    async with client.stream(
                        method=request.method,
                        url=target_url,
                        headers=headers,
                        params=params,
                        content=body_bytes
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk

                return StreamingResponse(
                    stream_response(),
                    media_type="application/json"
                )
            else:
                # For regular responses
                response = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    params=params,
                    content=body_bytes
                )

                # Create FastAPI response with same status code and headers
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.headers.get("content-type")
                )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error proxying request to model server: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))