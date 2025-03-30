# OsireLLM Testing Strategy and Plan

This document outlines the strategy and plan for implementing comprehensive tests for the OsireLLM codebase.

## Testing Framework & Setup

*   **Framework:** `pytest`
*   **Libraries:** `pytest-asyncio`, `pytest-mock` (or `unittest.mock`)
*   **Directory:** `/tests` at the project root.
*   **Dependencies:** Add testing libraries to development dependencies (e.g., `requirements-dev.txt`).
*   **Configuration:** Create basic `pytest` configuration if needed (`pytest.ini` or `pyproject.toml`).

## Attack Plan by Layer

### Testing Framework Setup [COMPLETE]
*   **Subtask:** Add `pytest`, `pytest-asyncio`, and `pytest-mock` to development dependencies. [COMPLETE]
*   **Subtask:** Create `/tests` directory. [COMPLETE]
*   **Subtask:** Create basic `pytest` configuration if needed. [COMPLETE]

### Task 1: Test Core Utilities & Configuration (Layer 1) [COMPLETE]
*   **File:** `tests/core/test_settings.py`
    *   **Subtask:** Test loading settings from dummy YAML. [COMPLETE]
    *   **Subtask:** Test environment variable overrides. [COMPLETE]
    *   **Subtask:** Test `base_url` computed field logic. [COMPLETE]
    *   **Subtask:** Test default values. [COMPLETE]
*   **File:** `tests/core/test_shell_commands.py`
    *   **Subtask:** Test `run_async_command` (mock asyncssh). [COMPLETE]
    *   **Subtask:** Test SLURM command wrappers (`run_async_sbatch`, etc.) verify calls. [COMPLETE]
    *   **Subtask:** Test `setup_ssh_keys_for_local_access` (mock os/subprocess). [COMPLETE]

### Task 2: Test Core Engine Logic (Layer 2)
*   **File:** `tests/core/engines/test_factory.py`
    *   **Subtask:** Implement tests for `validate_launch_args` (mocking engine validation). [RELEVANT]
    *   **Subtask:** Test `generate_script`. [RELEVANT]
    *   **Subtask:** Test `submit_launch`. [RELEVANT]
    *   **Subtask:** Test `get_manager_instance`. [RELEVANT]
*   **File:** `tests/core/engines/test_vllm.py` (and others)
    *   **Subtask:** Test engine `get_status`. [RELEVANT]
    *   **Subtask:** Test engine `terminate`. [RELEVANT]
    *   **Subtask:** Test engine static factory methods. [RELEVANT]

### Task 3: Test Core Resource Management (Layer 3)
*   **File:** `tests/core/test_resource_manager.py`
    *   **Subtask:** Implement `JobStateManager` fixtures with mocks. [RELEVANT]
    *   **Subtask:** Test `acquire_port`. [RELEVANT]
    *   **Subtask:** Test `add_job`, `get_job`, `get_all_jobs`. [RELEVANT]
    *   **Subtask:** Test `remove_job` (static/non-static). [RELEVANT]
    *   **Subtask:** Test `update_job_status`. [RELEVANT]
    *   **Subtask:** Test `terminate_job` (static/non-static, 403 error). [RELEVANT]
    *   **Subtask:** Test background tasks (may defer complexity). [PARTIALLY RELEVANT]
    *   **Subtask:** Design basic tests for locking (may defer complexity). [PARTIALLY RELEVANT]

### Task 4: Test Core Service Layer (Layer 4)
*   **File:** `tests/core/test_osire_llm_service.py`
    *   **Subtask:** Test `launch_job` scenarios (mock resource_manager/engine_factory). [RELEVANT]
    *   **Subtask:** Test `refresh_api_docs_with_model` (mock dependencies). [RELEVANT]
    *   **Subtask:** Test `has_docs_been_refreshed`. [RELEVANT]

### Task 5: Test API Routes Layer (Layer 5)
*   **File:** `tests/routes/test_osireLLM_routes.py`
    *   **Subtask:** Setup `TestClient` fixture. [RELEVANT]
    *   **Subtask:** Test `/launch` endpoint (mock service). [RELEVANT]
    *   **Subtask:** Test `/status` endpoint (mock resource_manager). [RELEVANT]
    *   **Subtask:** Test `/models` endpoint (mock resource_manager). [RELEVANT]
    *   **Subtask:** Test `/terminate/{model_name}` endpoint (mock resource_manager). [RELEVANT]

### Task 6: Test Application Layer (Layer 6)
*   **File:** `tests/test_main.py`
    *   **Subtask:** Setup `TestClient` fixture for `app.main.app`. [RELEVANT]
    *   **Subtask:** Test `startup_event` logic (mock dependencies). [RELEVANT]
    *   **Subtask:** Test `shutdown_event` logic (mock dependencies). [RELEVANT]

### Task 7: Define Strategy for System Tests (Layer 7)
*   **Subtask:** Document strategy and tools for system tests (e.g., Docker, test Slurm partition). [RELEVANT]