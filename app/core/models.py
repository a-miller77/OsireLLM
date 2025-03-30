from pydantic import BaseModel, Field, field_validator, PrivateAttr, ConfigDict, create_model
import os
from typing import Optional, List, Dict, Type, get_type_hints, Any
from enum import Enum
from datetime import datetime
import inspect
import logging

logger = logging.getLogger(__name__)

class BaseConfig(BaseModel):
    """Base configuration with common validation methods"""
    model_config = ConfigDict(validate_assignment=True)

class VLLMConfig(BaseConfig):
    """Configuration for a vLLM server"""
    model_name: str = Field(
        pattern="^[a-zA-Z0-9-_/.]+$",
        description="HuggingFace model name"
    )
    max_num_batched_tokens: int = Field(
        default=2048,
        gt=0,
        description="Maximum number of tokens that can be processed in a batch"
    )
    gpu_memory_utilization: float = Field(
        default=0.90,
        gt=0.0,
        le=1.0,
        description="Target GPU memory utilization"
    )
    dtype: str = Field(
        default="half",
        pattern="^(half|float|float16|float32)$",
        description="Data type for model weights"
    )
    max_model_len: int = Field(default=2048, gt=0)
    # Directory where models will be downloaded
    __download_dir: str = PrivateAttr(
        default="/data/ai_club/RosieLLM/models", #TODO update to Osire
    )
    vllm_extra_args: Optional[dict] = None

    @property
    def download_dir(self) -> str:
        return self.__download_dir

class OutputConfig(BaseModel):
    base_dir: str = Field(
        default="/data/ai_club/RosieLLM/out", #TODO update to Osire
        description="Base directory for output files"
    )
    stdout_file: Optional[str] = None
    stderr_file: Optional[str] = None
    log_level: str = Field(
        default="info",
        pattern="^(debug|info|warning|error|critical|trace)$"
    )

class SlurmConfig(BaseConfig):
    """Configuration for the SLURM aspect of a vLLM job"""
    job_name: str = "OsireLLM"
    partition: str = Field(
        default="teaching",
        pattern="^(teaching|highmem|dgx|dgxh100)$",
        description="SLURM partition to use"
    )
    nodes: int = Field(default=1, gt=0, le=2) #NOTE, this locks people to a maximum of 2 nodes
    gpus: int = Field(default=2, gt=0, le=8)
    cpus_per_gpu: int = Field(default=4, ge=2, le=16)
    time_limit: str = Field(
        default="3:00:00",
        pattern="^\d+:\d{2}:\d{2}$",
        description="Job time limit in HH:MM:SS format"
    )

    __output_config: OutputConfig = PrivateAttr(
        default_factory=lambda: OutputConfig(
            stdout_file=f"/data/ai_club/RosieLLM/out/{os.environ['USER']}_out.txt", #TODO update to Osire
            stderr_file=f"/data/ai_club/RosieLLM/out/{os.environ['USER']}_err.txt" #TODO update to Osire
        )
    )

    # Path to the container image
    __container: str = PrivateAttr(
        default="/data/ai_club/RosieLLM/RosieLLM.sif", #TODO update to Osire
    )
    # Volume mounts for the container
    __container_mounts: List[str] = PrivateAttr(
        default=["/data:/data"],
    )
    # Environment variables for the container
    __container_env: Dict[str, str] = PrivateAttr(
        default_factory=dict,
    )
    slurm_extra_args: Optional[dict] = None

    @property
    def output_config(self) -> OutputConfig:
        return self.__output_config

    @property
    def container_mounts(self) -> List[str]:
        return self.__container_mounts

    @property
    def container_env(self) -> Dict[str, str]:
        return self.__container_env

    @property
    def container(self) -> str:
        return self.__container

    @field_validator('gpus')
    def validate_gpus_per_partition(cls, v, info):
        partition = info.data.get('partition')
        if partition in ['teaching', 'highmem'] and v > 4:
            raise ValueError(f"Partition {partition} only has 4 GPUs per node available. Requested: {v}")
        return v

class JobState(str, Enum):
    PREEMPTED = "PREEMPTED" #finished
    PENDING = "PENDING" #active
    STARTING = "STARTING"  # When SLURM says running but API not up
    RUNNING = "RUNNING" #active
    COMPLETING = "COMPLETING" #finished
    COMPLETED = "COMPLETED" #finished
    CANCELLED = "CANCELLED" #finished
    FAILED = "FAILED" #finished
    SUSPENDED = "SUSPENDED" #finished
    STOPPED = "STOPPED" #finished
    UNKNOWN = "UNKNOWN"

    @property
    def is_active(self) -> bool:
        """Return True if the job is in an active state"""
        return self in [JobState.PENDING, JobState.STARTING, JobState.RUNNING]

    @property
    def is_finished(self) -> bool:
        """Return True if the job is in a terminal state"""
        return self in [
            JobState.CANCELLED, JobState.FAILED, JobState.COMPLETING,
            JobState.COMPLETED, JobState.SUSPENDED, JobState.STOPPED,
            JobState.PREEMPTED
        ]

class JobStatus(BaseModel):
    """Status information for a vLLM job"""
    job_id: str #SLURM id
    status: JobState #Using JobState enum
    model_name: str #LLM huggingface name
    num_gpus: int
    engine_type: str #vLLM or NIM
    partition: str #SLURM partition
    #vram: int #needed?
    node: Optional[str] = Field(
        None,
        pattern="^dh-[\w\d-]+$",
        description="SLURM node name"
    )
    port: Optional[int] = Field(
        None,
        ge=1024,
        le=65535,
        description="Port number for the vLLM server"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    owner: str = Field(
        description="Username who created the job",
        default_factory=lambda: os.environ['USER']
    )
    error_message: Optional[str] = None
    is_static: bool = False # Flag for models not managed by Slurm

    @property
    def server_url(self) -> Optional[str]:
        """Generate the server URL from node and port information"""
        if self.node and self.port:
            if not self.status == JobState.RUNNING:
                logger.warning(f"Server is not active, will not recieve requests")
            return f"http://{self.node}.hpc.msoe.edu:{self.port}"
        return None

    def model_dump(self, **kwargs):
        """Override model_dump to convert datetime objects to ISO format strings"""
        data = super().model_dump(**kwargs)
        # Convert datetime objects to ISO format strings
        if data.get('created_at'):
            data['created_at'] = data['created_at'].isoformat()
        if data.get('updated_at') and isinstance(data['updated_at'], datetime):
            data['updated_at'] = data['updated_at'].isoformat()
        return data

class LaunchRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to launch (e.g., meta-llama/Llama-2-7b-chat-hf)")
    engine_type: str = Field(..., description="Type of inference engine to use (e.g., 'vllm', 'nim')")
    engine_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary containing engine-specific configuration arguments (e.g., Slurm parameters, vLLM parameters)"
    )

class CreateJobRequest(BaseModel):
    model_name: str
    num_gpus: int
    partition: str

class AdminActionRequest(BaseModel):
    action: str
    model_name: Optional[str] = None


# KEEP THIS COMMENTED FOR NOW. IT MIGHT BE USEFUL IN THE FUTURE.
# def create_generic_launch_config() -> Type[BaseModel]:
#     """Dynamically generate GenericLaunchConfig from VLLMConfig and SlurmConfig"""

#     def get_field_info(model_class: Type[BaseModel]) -> dict:
#         """Extract field information from a Pydantic model class"""
#         fields = {}
#         for name, field in model_class.model_fields.items():
#             # Skip private fields (starting with _)
#             if name.startswith('_'):
#                 continue

#             # Get field info
#             field_info = {
#                 'type': field.annotation,
#                 'default': field.default,
#                 'description': field.description,
#             }

#             # Add any validators/constraints
#             for validator_name, value in field.json_schema_extra.items() if field.json_schema_extra else {}:
#                 if validator_name in ('gt', 'ge', 'lt', 'le', 'pattern'):
#                     field_info[validator_name] = value

#             fields[name] = (
#                 field_info['type'],
#                 Field(
#                     default=field_info['default'],
#                     description=field_info['description'],
#                     **{k: v for k, v in field_info.items()
#                        if k not in ('type', 'default', 'description')}
#                 )
#             )

#         return fields

#     # Get fields from both configs
#     vllm_fields = get_field_info(VLLMConfig)
#     slurm_fields = get_field_info(SlurmConfig)

#     # Check for duplicate fields
#     duplicates = set(vllm_fields.keys()) & set(slurm_fields.keys())
#     if duplicates:
#         logger.warning(
#             f"Found duplicate field names in configs: {duplicates}. "
#             "SlurmConfig fields will override VLLMConfig fields."
#         )

#     # Combine fields
#     all_fields = {
#         **vllm_fields,
#         **slurm_fields,
#     }

#     # Create and return the dynamic model
#     return create_model(
#         'GenericLaunchConfig',
#         __base__=BaseConfig,
#         __module__=__name__,
#         __doc__="Combined configuration for both vLLM and SLURM settings",
#         **all_fields
#     )

# # Create the GenericLaunchConfig class
# GenericLaunchConfig = create_generic_launch_config()