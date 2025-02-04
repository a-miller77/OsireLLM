from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from core.resource_manager import resource_manager

from routes import example, RosieLLM

# Create an instance of the APIRouter class
api_router = APIRouter()

# This is both a GET and a POST because the Rosie OOD performs a POST request by
# default to supply the API token in the body, not in the query parameters, including
# GET requests allows for this to be flexible for if the user refreshes the page.
@api_router.api_route("/", methods=["GET", "POST"], include_in_schema=False)
async def root() -> JSONResponse:
    return JSONResponse(
        status_code=200, content={"message": f"Welcome to the Rosie FastAPI Template"}
    )


# Include the router from routes/etc.py
api_router.include_router(example.router)
api_router.include_router(RosieLLM.router)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Start background tasks on app startup"""
    await resource_manager.start_cleanup_task()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up background tasks on app shutdown"""
    await resource_manager.stop_cleanup_task()
