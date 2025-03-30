import logging
from typing import Dict, Any, Tuple, Optional, Type

# Core imports
from ..models import JobStatus, VLLMConfig, SlurmConfig # Base models needed

# Engine imports
from .base import InferenceEngine
from .vllm import VllmEngine
from .nim import NimEngine

logger = logging.getLogger(__name__)

# --- Engine Mapping ---
# Maps engine_type string to the corresponding engine class
ENGINE_CLASS_MAP: Dict[str, Type[InferenceEngine]] = {
    "vllm": VllmEngine,
    "nim": NimEngine,
    # Add mappings for new engines here
}

# --- Factory Functions ---

def get_engine_class(engine_type: str) -> Type[InferenceEngine]:
    """Get the engine class corresponding to the engine_type string."""
    engine_cls = ENGINE_CLASS_MAP.get(engine_type.lower())
    if not engine_cls:
        logger.error(f"Unsupported engine type requested: {engine_type}")
        raise ValueError(f"Unsupported engine type: {engine_type}")
    return engine_cls

# --- Launch Dispatch Functions ---

def validate_launch_args(engine_type: str, args: Dict[str, Any]) -> Tuple[Any, SlurmConfig]:
    """
    Dispatch validation to the appropriate engine's static method.

    Returns:
        Tuple containing validated configuration objects (e.g., (VLLMConfig, SlurmConfig)).
        The exact types depend on the engine implementation.
    """
    logger.debug(f"Dispatching launch arg validation for engine type: {engine_type}")
    engine_cls = get_engine_class(engine_type)
    # Assumes a static method 'validate_launch_args' exists on the class
    if not hasattr(engine_cls, 'validate_launch_args'):
        raise NotImplementedError(f"Engine type '{engine_type}' does not support launch argument validation.")
    # The return type here is tricky to annotate perfectly without Generics/Protocols,
    # but it should match the return type of the specific engine's validator.
    return engine_cls.validate_launch_args(args) # type: ignore

def generate_script(engine_type: str, model_name: str, port: int, *config_objs: Any) -> Tuple[str, str]:
    """
    Dispatch script generation to the appropriate engine's static method.

    Args:
        engine_type: The type of engine.
        model_name: The name of the model being launched.
        port: The allocated port for the server.
        *config_objs: Validated configuration objects returned by validate_launch_args.

    Returns:
        Tuple[str, str]: (script_content, job_name)
    """
    logger.debug(f"Dispatching script generation for engine type: {engine_type}")
    engine_cls = get_engine_class(engine_type)
    if not hasattr(engine_cls, 'generate_script'):
        raise NotImplementedError(f"Engine type '{engine_type}' does not support script generation.")

    # Specific handling for known engines requiring specific config types
    if engine_type == "vllm":
        if len(config_objs) == 2 and isinstance(config_objs[0], VLLMConfig) and isinstance(config_objs[1], SlurmConfig):
            return engine_cls.generate_script(config_objs[0], config_objs[1], port, model_name)
        else:
            raise TypeError(f"Incorrect config objects provided for vLLM engine script generation.")
    # Add cases for other engines here
    # else:
    #     # Generic call, assuming signature matches or engine handles it
    #     return engine_cls.generate_script(*config_objs, port=port, model_name=model_name)
    else:
         # Fallback or raise error for unhandled types
         raise NotImplementedError(f"Script generation dispatch not implemented for engine type '{engine_type}' with provided config.")


async def submit_launch(engine_type: str, script_content: str, job_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Dispatch launch command submission to the appropriate engine's static method.
    """
    logger.debug(f"Dispatching launch submission for engine type: {engine_type}")
    engine_cls = get_engine_class(engine_type)
    if not hasattr(engine_cls, 'submit_launch_command'):
         raise NotImplementedError(f"Engine type '{engine_type}' does not support launch submission.")
    return await engine_cls.submit_launch_command(script_content, job_name)


# --- Management Instance Factory ---

def get_manager_instance(job_status: JobStatus) -> InferenceEngine:
    """
    Get an instance of the appropriate InferenceEngine subclass for managing
    an existing job based on its status.
    """
    engine_type = job_status.engine_type
    logger.debug(f"Getting manager instance for job {job_status.job_id} (engine type: {engine_type})")
    engine_cls = get_engine_class(engine_type)
    return engine_cls(job_status)