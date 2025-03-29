from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

# route imports
from routes import model_proxy
from routes import OsireLLM
from routes import Admin  # Add import for the new Admin router

# Create an instance of the APIRouter class
api_router = APIRouter()

# This is both a GET and a POST because the Rosie OOD performs a POST request by
# default to supply the API token in the body, not in the query parameters, including
# GET requests allows for this to be flexible for if the user refreshes the page.
@api_router.api_route("/", methods=["GET", "POST"], include_in_schema=False)
async def root() -> JSONResponse:
    return JSONResponse(
        status_code=200, content={"message": f"Welcome to the OsireLLM Orchestrator"}
    )

# Include the router from routes/etc.py
api_router.include_router(OsireLLM.router)
api_router.include_router(Admin.router)
api_router.include_router(model_proxy.router) #NOTE: must be last

# app = FastAPI()

# @app.on_event("startup")
# async def startup_event():
#     """Start background tasks on app startup"""
#     logger.info("Starting application background tasks")
#     await resource_manager.start_cleanup_task()

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Clean up background tasks and cancel all jobs on app shutdown"""
#     logger.info("Starting application shutdown")
#     try:
#         # Cancel all running jobs
#         await resource_manager.cancel_all_jobs()
#         # Stop the cleanup task
#         await resource_manager.stop_cleanup_task()
#         logger.info("Application shutdown completed successfully")
#     except Exception as e:
#         logger.error(f"Error during shutdown: {str(e)}")
