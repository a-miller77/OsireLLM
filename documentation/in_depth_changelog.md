## Task 1: Configuration Management Refactoring (Completed 2025-03-29)

*   **Feature:** Implemented a centralized configuration system using Pydantic `BaseSettings` and a YAML file.
*   **Details:**
    *   Created `app/core/osire_config.yaml` to store application settings like server details, SLURM parameters, SSH settings, static model definitions, and background task intervals.
    *   Modified `app/core/settings.py` to load configuration from `osire_config.yaml`, `.env` files, and environment variables, establishing a clear precedence order (Defaults < YAML < .env < Env Vars). Defined nested Pydantic models (`ServerConfig`, `SlurmConfig`, etc.) for validation and structure.
    *   Replaced numerous hardcoded values throughout the `app/core/` directory (`resource_manager.py`, `shell_commands.py`) and `app/main.py` with references to the unified `settings` object.
    *   Refactored the handling of pre-existing, non-SLURM models (previously hardcoded DGX model) into a configurable list of `static_models` within `osire_config.yaml`. Each static model can now have its own configuration (URL, health endpoint, metadata).
    *   Updated `app/core/resource_manager.py` to dynamically check and manage these configured static models during startup and status updates.
    *   Added an `is_static` flag to the `app/core/models.py::JobStatus` model to distinguish between Slurm-managed jobs and statically configured services.

## Task 2: Engine Logic Separation (Completed 2025-03-29)

*   **Goal:** Refactored core logic to separate concerns for different backend inference engines (vLLM, NIM), making the `ResourceManager` engine-agnostic.
*   **Key Changes:**
    *   Created new `app/core/engines/` directory to house engine-specific implementations.
    *   Defined an abstract `InferenceEngine` base class (`app/core/engines/base.py`) specifying the interface for job management (status checks, termination).
    *   Implemented `VllmEngine` (`app/core/engines/vllm.py`) containing logic for Slurm script generation, job submission (`sbatch`), status checking (`scontrol`, `sacct`, HTTP health), and termination (`scancel`).
    *   Implemented `NimEngine` (`app/core/engines/nim.py`) as a stub, primarily handling status checks for statically configured NIM instances via HTTP health checks. Launch and termination are marked as `NotImplementedError`.
    *   Introduced an engine factory (`app/core/engines/factory.py`) to dispatch calls (validation, script generation, submission, instance management) to the correct engine class based on `engine_type`.
    *   Refactored `app/core/resource_manager.py`:
        *   Removed all engine-specific logic (Slurm commands, script generation, vLLM health checks).
        *   Updated `update_job_status` and `terminate_job` to use the engine factory to get an engine instance and delegate the actual work.
        *   Reviewed and standardized locking (`_dict_lock`, `_model_locks`) for dictionary access and per-job operations.
        *   Removed the `launch_job` method (moved orchestration to service layer).
    *   Refactored `app/core/osire_llm_service.py`:
        *   Removed old engine-specific functions (`_generate_slurm_script`, `launch_server`, `split_config`).
        *   Implemented the `launch_job` orchestration method, coordinating calls between the `engine_factory` and `resource_manager`.
    *   Refactored the `/launch` API endpoint (`app/routes/osireLLM.py`) to accept the new `LaunchRequest` model and call the `osire_llm_service.launch_job` function.
    *   Updated `app/core/models.py` with `LaunchRequest` and added `engine_type` to `JobStatus`.

## [Unreleased] - 3/29/2025

### Added
- **Testing (Task 1):**
  - Created test suite for `app/core/settings.py` (`tests/core/test_settings.py`):
    - Test loading settings from YAML.
    - Test environment variable overrides (using `__` delimiter).
    - Test specific `BASE_URL` environment variable override.
    - Test default values when settings are missing.
  - Created test suite for `app/core/shell_commands.py` (`tests/core/test_shell_commands.py`):
    - Test `run_async_command` for success, command failure, and SSH connection errors (mocking `asyncssh`).
    - Test SLURM wrappers (`run_async_sbatch`, `run_async_scancel`, `run_async_scontrol`, `run_async_sacct`) ensure correct calls to `run_async_command` and test output parsing.
    - Test `setup_ssh_keys_for_local_access` for key existence, generation success/failure (mocking `os`, `subprocess`).
    - Test `add_key_to_authorized_keys` helper function.

### Changed
- N/A