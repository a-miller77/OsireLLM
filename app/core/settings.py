import os
from functools import lru_cache
from typing import Annotated

from pydantic import BeforeValidator
from pydantic_settings import BaseSettings

# Get the absolute path of the .env file
abs_path_env = os.path.abspath("../../.env")

# Define a validator function to strip whitespace
def strip_whitespace(v: str) -> str:
    return v.strip() if isinstance(v, str) else v

# Type for string fields that should have whitespace stripped
StrippedStr = Annotated[str, BeforeValidator(strip_whitespace)]

# The Settings class which will retrieve the environment variables
class Settings(BaseSettings):
    API_TOKEN: StrippedStr
    SALT: StrippedStr

    # Feel free to modify these for your platform
    APP_NAME: str = "Rosie FastAPI Template"
    APP_VERSION: str = "0.0.0"
    APP_DESC: str = (
        "A FastAPI template for building APIs on Rosie, the MSOE supercomputer. \nDeveloped by: Adam Haile - 2024"
    )

    ENVIRONMENT: str = "Rosie"
    DEVICE: str = "cuda"
    BASE_URL: str = ""

    class Config:
        env_file = ".env"


# Cache the settings to avoid reading the .env file multiple times
@lru_cache()
def get_settings() -> Settings:
    return Settings()
