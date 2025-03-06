from typing import Tuple, Dict, Any
import logging
import os
import random
import subprocess as sp
import pydantic
from pydantic import Field
import fabric
from fabric import Connection
from invoke import UnexpectedExit
import asyncssh

# Configure logging
logger = logging.getLogger(__name__)
async_ssh_logger = logging.getLogger("asyncssh")
async_ssh_logger.setLevel(logging.WARNING)

MGMT_NODE_COUNT = 4
MGMT_NODE_PATTERN = "dh-mgmt{}.hpc.msoe.edu"

# Default connection settings
class SSHConfig(pydantic.BaseModel):
    host: str = Field(default_factory=lambda: 
        MGMT_NODE_PATTERN.format(random.randint(1, MGMT_NODE_COUNT))
    )
    user: str = Field(default_factory=lambda: os.environ.get("USER", ""))
    connect_kwargs: Dict[str, Any] = Field(default_factory=dict)

SSH_CONFIG = SSHConfig()    

def get_connection(host: str = SSH_CONFIG.host, 
                  user: str = SSH_CONFIG.user, 
                  **kwargs) -> Connection:
    """Get a Fabric SSH connection with specified or default parameters."""
    config = SSH_CONFIG.dict()
    if host != SSH_CONFIG.host:
        config["host"] = host
    if user != SSH_CONFIG.user:
        config["user"] = user
    if kwargs:
        for key, value in kwargs.items():
            config[key] = value
    
    return Connection(**config)

def run_command(command: str, 
               host: str = SSH_CONFIG.host,
               user: str = SSH_CONFIG.user,
               capture_output: bool = True,
               **kwargs) -> Tuple[str, str, int]:
    """
    Run a shell command via SSH using Fabric.
    Returns: Tuple of (stdout, stderr, return_code)
    """
    logger.debug(f"Running command: {command}")
    conn = get_connection(host, user, **kwargs)
    
    try:
        result = conn.run(command, warn=True, hide=capture_output)
        return result.stdout, result.stderr, result.return_code
    except UnexpectedExit as e:
        logger.error(f"Command failed: {e}")
        return "", str(e), 1
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return "", str(e), 1
    finally:
        conn.close()

def run_sbatch(script_path: str, 
              host: str = SSH_CONFIG.host,
              user: str = SSH_CONFIG.user) -> Tuple[str, str, int]:
    """Run an sbatch command to submit a SLURM job."""
    logger.info(f"Submitting SLURM job: {script_path}")
    return run_command(f"sbatch {script_path}", host, user)

def run_scancel(job_id: str,
               host: str = SSH_CONFIG.host,
               user: str = SSH_CONFIG.user) -> Tuple[str, str, int]:
    """Cancel a SLURM job via scancel."""
    logger.info(f"Canceling SLURM job {job_id}")
    return run_command(f"scancel {job_id}", host, user)

def run_scontrol(job_id: str,
                host: str = SSH_CONFIG.host,
                user: str = SSH_CONFIG.user) -> Tuple[Dict, str, int]:
    """Get detailed information about a SLURM job via scontrol."""
    logger.debug(f"Getting info for SLURM job {job_id}")
    stdout, stderr, return_code = run_command(f"scontrol show job {job_id}", host, user)
    
    if return_code != 0:
        logger.error(f"Failed to get job info: {stderr}")
        return {}, stderr, return_code
    
    # Parse scontrol output into a dictionary
    info = {}
    for item in stdout.split():
        if '=' in item:
            key, value = item.split('=', 1)
            info[key] = value
    
    return info, stderr, return_code

# Async versions of the commands

async def get_async_connection(
    host: str = SSH_CONFIG.host, 
    user: str = SSH_CONFIG.user,
    **kwargs
) -> asyncssh.SSHClientConnection:
    """Get an asyncssh connection with specified or default parameters."""
    try:
        options = {"known_hosts": None}  # Disable known_hosts checking
        options.update(kwargs)
        
        conn = await asyncssh.connect(host, username=user, **options)
        return conn
    except Exception as e:
        logger.error(f"Failed to establish SSH connection: {e}")
        raise

async def run_async_command(
    command: str, 
    host: str = SSH_CONFIG.host,
    user: str = SSH_CONFIG.user,
    **kwargs
) -> Tuple[str, str, int]:
    """Run a shell command asynchronously via SSH."""
    logger.debug(f"Running async command: {command}")
    
    try:
        conn = await get_async_connection(host, user, **kwargs)
        try:
            process = await conn.run(command, check=False)
            return process.stdout, process.stderr, process.exit_status
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error in async command execution: {e}")
        return "", str(e), 1

async def run_async_sbatch(script_path: str, 
                          host: str = SSH_CONFIG.host,
                          user: str = SSH_CONFIG.user) -> Tuple[str, str, int]:
    """Run an sbatch command asynchronously."""
    logger.info(f"Submitting async SLURM job: {script_path}")
    return await run_async_command(f"sbatch {script_path}", host, user)

async def run_async_scancel(job_id: str,
                           host: str = SSH_CONFIG.host,
                           user: str = SSH_CONFIG.user) -> Tuple[str, str, int]:
    """Cancel a SLURM job asynchronously."""
    logger.info(f"Canceling async SLURM job {job_id}")
    return await run_async_command(f"scancel {job_id}", host, user)

async def run_async_scontrol(job_id: str,
                            host: str = SSH_CONFIG.host,
                            user: str = SSH_CONFIG.user) -> Tuple[Dict, str, int]:
    """Get information about a SLURM job asynchronously."""
    logger.debug(f"Getting async info for SLURM job {job_id}")
    stdout, stderr, return_code = await run_async_command(f"scontrol show job {job_id}", host, user)
    
    if return_code != 0:
        logger.error(f"Failed to get async job info: {stderr}")
        return {}, stderr, return_code
    
    # Parse scontrol output into a dictionary
    info = {}
    for item in stdout.split():
        if '=' in item:
            key, value = item.split('=', 1)
            info[key] = value
    
    return info, stderr, return_code

async def setup_ssh_keys_for_local_access(
    key_name: str = "id_rsa_osire", 
    key_dir: str = "~/.ssh"
) -> Tuple[bool, str]:
    """Set up SSH keys for HPC cluster access."""
    # Expand the path if it contains ~
    key_dir = os.path.expanduser(key_dir)
    
    # Create SSH directory if it doesn't exist
    if not os.path.exists(key_dir):
        os.makedirs(key_dir, mode=0o700)
    
    key_path = os.path.join(key_dir, key_name)
    pub_key_path = f"{key_path}.pub"
    
    # Check if keys already exist
    if os.path.exists(key_path) and os.path.exists(pub_key_path):
        logger.info(f"SSH keys already exist at {key_path}")
        return True, f"SSH keys already exist at {key_path}"
    
    # Generate new SSH key pair - use local subprocess, not SSH
    logger.info(f"Generating new SSH key pair at {key_path}")
    cmd = f"ssh-keygen -t rsa -b 4096 -f {key_path} -N '' -C 'osire-hpc-access'"
    
    try:
        # Run locally using subprocess, not via SSH
        process = sp.run(
            cmd, 
            shell=True, 
            check=True, 
            text=True,
            capture_output=True
        )
        
        # Add key to authorized_keys
        auth_keys_path = os.path.join(key_dir, "authorized_keys")
        
        # Read public key
        with open(pub_key_path, 'r') as f:
            pub_key = f.read().strip()
        
        # Check if key already in authorized_keys
        key_added = False
        if os.path.exists(auth_keys_path):
            with open(auth_keys_path, 'r') as f:
                if pub_key in f.read():
                    key_added = True
        
        # Add key if not already added
        if not key_added:
            with open(auth_keys_path, 'a') as f:
                f.write(f"\n{pub_key}\n")
            
            # Set proper permissions
            os.chmod(auth_keys_path, 0o600)
        
        # Update SSH config to use this key for management nodes
        config_path = os.path.join(key_dir, "config")
        config_entry = f"""
# Added by OsireLLM for HPC cluster connections
Host dh-mgmt*.hpc.msoe.edu
    IdentityFile {key_path}
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
"""
        
        # Check if config exists and if entry already present
        config_updated = False
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_content = f.read()
                if "Added by OsireLLM for HPC cluster connections" in config_content:
                    config_updated = True
        
        # Add config entry if not already present
        if not config_updated:
            with open(config_path, 'a') as f:
                f.write(config_entry)
            
            # Set proper permissions
            os.chmod(config_path, 0o600)
        
        # Update the SSH connection config to use the new key
        update_ssh_connection_config(use_key=True, key_name=key_name)
        
        return True, f"SSH keys successfully set up at {key_path}"
        
    except sp.CalledProcessError as e:
        logger.error(f"Error generating SSH keys: {e.stderr}")
        return False, f"Failed to generate SSH keys: {e.stderr}"
    except Exception as e:
        logger.error(f"Error setting up SSH keys: {e}")
        return False, f"Failed to set up SSH keys: {e}"

def get_ssh_key_path(key_name: str = "id_rsa_osire") -> str:
    """Get the path to the SSH key used for local connections."""
    home_dir = os.path.expanduser("~")
    key_dir = os.path.join(home_dir, ".ssh")
    return os.path.join(key_dir, key_name)

def update_ssh_connection_config(
    use_key: bool = True, 
    key_name: str = "id_rsa_osire"
) -> None:
    """Update the SSH connection configuration to use the specific keys."""
    if use_key:
        key_path = get_ssh_key_path(key_name)
        
        # Update SSH config to use this key and ignore host checking
        SSH_CONFIG.connect_kwargs = {
            "key_filename": key_path,
            "allow_agent": False,
            "look_for_keys": False,
        }
        
        logger.info(f"Updated SSH config to use key: {key_path}") 