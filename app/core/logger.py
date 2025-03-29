import logging
import sys
from pydantic import BaseModel

from .settings import get_settings

# Create the Colors object to add colors to the logs
if get_settings().ENVIRONMENT == "local":
    class COLORS(BaseModel):
        DEBUG: str = "\033[34m"  # Blue
        INFO: str = "\033[32m"  # Green
        WARNING: str = "\033[33m"  # Yellow
        ERROR: str = "\033[31m"  # Red
        CRITICAL: str = "\033[41m"  # Red Background
        RESET: str = "\033[0m"
else:
    class COLORS(BaseModel):
        DEBUG: str = ""
        INFO: str = ""
        WARNING: str = ""
        ERROR: str = ""
        CRITICAL: str = ""
        RESET: str = ""

# Set up root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear any existing handlers to avoid duplicates
if root_logger.handlers:
    root_logger.handlers.clear()

# Create console handler for logging to stdout (which gets captured by SLURM)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Create formatter
colors = COLORS()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)

# Add handler to root logger
root_logger.addHandler(console_handler)

# Force propagate logs from all libraries to the root logger
for name in logging.root.manager.loggerDict:
    logger = logging.getLogger(name)
    logger.propagate = True

# Set levels for specific loggers
logging.getLogger("core").setLevel(logging.INFO)
logging.getLogger("api").setLevel(logging.INFO)
logging.getLogger("routes").setLevel(logging.INFO)

# Make uvicorn use our logger
logging.getLogger("uvicorn").handlers = root_logger.handlers
logging.getLogger("uvicorn.access").handlers = root_logger.handlers
logging.getLogger("uvicorn.error").handlers = root_logger.handlers

# Disable propagation for uvicorn loggers to avoid duplicate logs
logging.getLogger("uvicorn").propagate = False
logging.getLogger("uvicorn.access").propagate = False
logging.getLogger("uvicorn.error").propagate = False

# Log configuration complete
root_logger.info("Logging configuration complete")
