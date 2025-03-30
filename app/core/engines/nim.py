import logging
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import requests

# Core imports
from ..models import JobStatus, JobState # Need these for status reporting
from ..settings import get_settings
from .base import InferenceEngine

logger = logging.getLogger(__name__)
settings = get_settings()

# Placeholder for NIM specific config if needed later
# class NimConfig(BaseModel): ...

class NimEngine(InferenceEngine):
    """
    InferenceEngine implementation stub for NVIDIA NIM models.
    Currently provides read-only status/schema checks and raises errors for launch/termination.
    Assumes NIM models are managed externally (like static models).
    """

    # --- Static Methods (Stubs) --- #

    @staticmethod
    def validate_launch_args(args: Dict[str, Any]) -> Tuple[Any, ...]:
        """NIM launch is not supported via OsireLLM at this time."""
        logger.error("Attempted to validate launch args for unsupported NIM engine.")
        raise NotImplementedError("Launching NIM models via OsireLLM is not currently supported.")

    @staticmethod
    def generate_script(*args, **kwargs) -> Tuple[str, str]:
        """NIM launch is not supported via OsireLLM at this time."""
        logger.error("Attempted to generate script for unsupported NIM engine.")
        raise NotImplementedError("Launching NIM models via OsireLLM is not currently supported.")

    @staticmethod
    async def submit_launch_command(*args, **kwargs) -> Tuple[Optional[str], Optional[str]]:
        """NIM launch is not supported via OsireLLM at this time."""
        logger.error("Attempted to submit launch command for unsupported NIM engine.")
        raise NotImplementedError("Launching NIM models via OsireLLM is not currently supported.")

    # --- Instance Methods (Read-Only Stubs) --- #

    def __init__(self, job_status: JobStatus):
        """
        Initialize the NIM engine instance for managing an existing (externally managed) job.
        """
        super().__init__(job_status)
        # Find the corresponding static config for this NIM service if needed for URLs/endpoints
        # This assumes 'nim' is used as engine_type in the static_models config
        self._static_config = next((m for m in settings.static_models if m.id == self.job_status.job_id and hasattr(m, 'engine_type') and m.engine_type == 'nim'), None)
        if not self._static_config:
             logger.warning(f"Could not find static_models config entry for NIM job ID {self.job_status.job_id}")

    async def _check_http_status(self) -> Tuple[bool, Optional[str]]:
        """Perform HTTP health check for a static NIM service.

        Returns:
            Tuple[bool, Optional[str]]: (is_healthy, error_message)
        """
        logger.debug(f"Checking HTTP status for static NIM job {self.job_status.job_id}")
        server_url = self.job_status.server_url
        error_message: Optional[str] = None

        # Use health_endpoint from static config if available, otherwise default
        health_endpoint = "/v1/health/ready" # Default NIM health endpoint?
        if self._static_config and hasattr(self._static_config, 'health_endpoint') and self._static_config.health_endpoint:
             health_endpoint = self._static_config.health_endpoint

        if not server_url:
             logger.warning(f"Cannot check NIM status for {self.job_status.job_id}: server_url is not set.")
             return False, "Server URL not configured"

        health_url = f"{server_url.rstrip('/')}/{health_endpoint.lstrip('/')}"
        is_healthy = False
        try:
            response = requests.get(health_url, timeout=2)
            is_healthy = 200 <= response.status_code < 300
            if not is_healthy:
                error_message = f"NIM service health check failed (Status: {response.status_code}) at {health_url}"
        except requests.Timeout: error_message = f"NIM service health check timed out at {health_url}"
        except requests.RequestException as e: error_message = f"NIM service health check connection failed at {health_url}: {e}"
        except Exception as e:
            error_message = f"Unexpected error during NIM health check at {health_url}: {e}"
            logger.error(error_message, exc_info=True)

        if error_message: logger.debug(error_message) # Log errors at debug level
        return is_healthy, error_message

    async def get_status(self) -> JobStatus:
        """
        Get the status of the NIM service. Currently only supports static services.
        Updates the internal job_status object.
        """
        updated_status = self.job_status.copy(deep=True)
        updated_status.updated_at = datetime.utcnow()
        updated_status.error_message = None

        if updated_status.is_static:
            # NIM is assumed to be static/external for now
            is_healthy, check_error = await self._check_http_status()
            updated_status.status = JobState.RUNNING if is_healthy else JobState.FAILED
            updated_status.error_message = check_error
        else:
            # Launching NIM via Slurm/other is not implemented
            logger.error(f"get_status called for non-static NIM job {self.job_status.job_id}, which is not supported.")
            updated_status.status = JobState.UNKNOWN
            updated_status.error_message = "Status check for dynamically launched NIM is not implemented."

        # Persist and return updated status
        self.job_status = updated_status
        return updated_status

    async def terminate(self) -> bool:
        """NIM termination is not supported via OsireLLM."""
        logger.error(f"Attempted to terminate externally managed NIM service {self.job_status.job_id}. Operation not supported.")
        raise NotImplementedError("Terminating NIM models via OsireLLM is not currently supported as they are externally managed.")

    async def get_schema_url(self) -> Optional[str]:
        """
        Get the OpenAPI schema URL for the NIM service.
        """
        # Assume standard schema path for NIM? Needs confirmation.
        schema_path = "/openapi.json" # Placeholder

        if self.job_status.status == JobState.RUNNING and self.job_status.server_url:
            schema_url = f"{self.job_status.server_url.rstrip('/')}{schema_path}"
            logger.debug(f"Schema URL for NIM service {self.job_status.job_id}: {schema_url}")
            return schema_url
        else:
            logger.debug(f"Cannot get schema URL for NIM service {self.job_status.job_id} (Status: {self.job_status.status}, URL: {self.job_status.server_url})")
            return None