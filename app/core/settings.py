import os
import yaml
from functools import lru_cache
from typing import Annotated, Optional, Dict, Any, List

from pydantic import BeforeValidator, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the absolute path of the .env file
abs_path_env = os.path.abspath("../../.env")

# Define a validator function to strip whitespace
def strip_whitespace(v: str) -> str:
    return v.strip() if isinstance(v, str) else v

# Type for string fields that should have whitespace stripped
StrippedStr = Annotated[str, BeforeValidator(strip_whitespace)]

# Define nested configuration models
class ServerConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    web_domain: str = "https://dh-ood.hpc.msoe.edu" # Do not change
    base_url_from_config: str = Field(default="/", alias='base_url')
    node_url_template: str = "{node_url}.hpc.msoe.edu:{port}"

    @computed_field
    @property
    def base_url(self) -> str:
        """Return BASE_URL from env var if set, otherwise return value from config."""
        return os.environ.get("BASE_URL", self.base_url_from_config)

class SlurmConfig(BaseSettings):
    output_dir: str = "/path/to/default/slurm/output"
    job_script_template: str = "/path/to/default/template.slurm"
    preferred_ports: List[int] = [8000, 7777, 8080]  # Default ports, more added in code

class SSHConfig(BaseSettings):
    management_node_count: int = 4
    management_node_pattern: str = "dh-mgmt{}.hpc.msoe.edu"
    key_name: str = "id_rsa_osire"
    key_dir: str = "~/.ssh"

# Define config for a single static model
class StaticModelConfig(BaseSettings):
    id: str # Internal identifier
    model_name: str # Name users request
    engine_type: str # Type of engine (e.g., 'vllm', 'nim')
    server_url: str
    node: Optional[str] = None # Informational
    port: Optional[int] = None # Informational
    num_gpus: Optional[int] = None # Informational
    partition: Optional[str] = None # Informational
    health_endpoint: str = "/health" # Default health endpoint

class JobStateManagerConfig(BaseSettings):
    cleanup_interval: int = 900
    update_interval: int = 300
    fast_update_interval: int = 3
    fast_update_backoff: int = 30
    fast_update_duration: int = 300

class AppConfig(BaseSettings):
    name: str = "OsireLLM"
    version: str = "0.1.0"
    description: str = "OsireLLM API for orchestrating LLM inference on HPC resources"

# The Settings class which will retrieve settings from YAML and environment variables
class Settings(BaseSettings):
    # Existing environment variables
    API_TOKEN: StrippedStr
    SALT: StrippedStr

    # Previous hardcoded app settings, now from config
    ENVIRONMENT: str = "Rosie"
    DEVICE: str = "cuda"

    # New configuration from YAML
    server: ServerConfig = Field(default_factory=ServerConfig)
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)
    ssh: SSHConfig = Field(default_factory=SSHConfig)
    static_models: List[StaticModelConfig] = [] # List of static models
    job_state_manager: JobStateManagerConfig = Field(default_factory=JobStateManagerConfig)
    app: AppConfig = Field(default_factory=AppConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",  # Use double underscore for nested settings in env vars (e.g., SERVER__HOST)
    )

    # Method to pre-populate settings from YAML before environment variables
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Load from YAML first
        yaml_settings = cls._load_yaml_settings()

        # Return the settings chaining YAML first, then the regular sources
        return (
            init_settings,            # 1. init_settings has highest priority
            env_settings,             # 2. environment variables
            dotenv_settings,          # 3. .env file
            lambda: yaml_settings,    # 4. WRAPPED yaml file settings source
            file_secret_settings,     # 5. file secrets (rarely used)
        )

    @classmethod
    def _load_yaml_settings(cls) -> Dict[str, Any]:
        """Load settings from the YAML configuration file."""
        try:
            yaml_path = os.path.join(os.path.dirname(__file__), "osire_config.yaml")
            with open(yaml_path, "r") as file:
                yaml_settings = yaml.safe_load(file) or {}

            # REMOVED Flattening - Pydantic handles nested models directly from nested dicts
            # return cls._flatten_dict(yaml_settings)
            return yaml_settings # Return the nested dict directly
        except Exception as e:
            print(f"Warning: Failed to load YAML settings: {e}")
            return {}

# Cache the settings to avoid reading the configuration files multiple times
@lru_cache()
def get_settings() -> Settings:
    return Settings()
