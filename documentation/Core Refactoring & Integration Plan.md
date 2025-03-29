# OsireLLM - Core Refactoring & Integration Plan

This plan outlines the steps to address Roadmap Item 1: "Core Refactoring & Integration" from the project's README.

## TASK 1: Configuration Management Refactoring

*   **Goal:** Move hardcoded configuration values to a central configuration mechanism (local file for now).
*   **Files:**
    *   `app/core/osire_config.yaml` (New file)
    *   `app/core/config.py` (New file)
    *   `app/core/resource_manager.py`
    *   `app/main.py`
    *   (Potentially others like `app/core/shell_commands.py` if they contain hardcoded values relevant to configuration)
*   **Subtasks:**
    *   [RELEVANT] [MINOR] Create `app/core/osire_config.yaml` and `app/core/config.py`. Define a structure/class within `app/core/config.py` to hold and load settings, loading them from `app/core/osire_config.yaml`.
    *   [RELEVANT] [MODERATE] Identify hardcoded values (e.g., SLURM paths, output directories, container locations, default ports, management node names) in relevant files. Add them to `app/core/osire_config.yaml`.
    *   [RELEVANT] [MODERATE] Replace identified hardcoded values with references to the settings defined in `app/core/config.py`.

## TASK 2: Engine Logic Separation

*   **Goal:** Refactor `resource_manager.py` to separate logic for different backend engines (vLLM, NIM).
*   **Files:**
    *   `app/core/resource_manager.py`
    *   `app/core/engines/` (New directory)
    *   `app/core/engines/__init__.py` (New file)
    *   `app/core/engines/base.py` (New file)
    *   `app/core/engines/vllm.py` (New file)
    *   `app/core/engines/nim.py` (New file)
    *   `app/core/models.py`
*   **Subtasks:**
    *   [RELEVANT] [MINOR] Create the directory `app/core/engines/` and an empty `__init__.py` file within it.
    *   [RELEVANT] [MODERATE] Define a base engine abstract class or interface in `app/core/engines/base.py` outlining common methods (e.g., `launch`, `get_status`, `terminate`, `get_schema_url`, `generate_job_script`).
    *   [RELEVANT] [MAJOR] Create specific engine classes (e.g., `VllmEngine`, `NimEngine`) in `app/core/engines/vllm.py` and `app/core/engines/nim.py` inheriting from the base. Move engine-specific logic (SLURM script variations, status commands, schema fetching logic) from `resource_manager.py` into these classes.
    *   [PARTIALLY RELEVANT] [MODERATE] Enhance `app/core/models.py`: Modify the `JobStatus` model to include an `engine_type` field or similar, and potentially engine-specific configuration details needed at runtime (e.g., container image, specific launch args). This supports DRY principle. Explanation: Storing engine type helps ResourceManager delegate correctly and avoids redundant checks.
    *   [RELEVANT] [MAJOR] Refactor `app/core/resource_manager.py` to instantiate and use the appropriate engine class based on the request or `JobStatus.engine_type`. Delegate engine-specific operations (like script generation, status checks) to the engine object. Consider using a Factory pattern here for cleaner engine instantiation.

## TASK 3: Service Registry Placeholder

*   **Goal:** Implement a basic mechanism to store service information locally.
*   **Files:**
    *   `app/core/service_registry.py` (New file)
    *   `app/main.py`
    *   (Potentially a new routes file like `app/routes/admin.py` if an endpoint is added)
*   **Subtasks:**
    *   [RELEVANT] [MINOR] Create `app/core/service_registry.py`.
    *   [RELEVANT] [MODERATE] Define a class or functions within `service_registry.py` to manage service state (e.g., store start time, status in memory). Include a function to register the OsireLLM service itself on startup.
    *   [RELEVANT] [MINOR] In `app/main.py`'s startup event handler, call the registration function from the new service registry module.
    *   [OPTIONAL] [MINOR] Add a simple API endpoint (e.g., `/admin/service-info`) to display the locally stored service information. Explanation: This provides a simple way to verify the placeholder registry is populated correctly during startup.

## TASK 4: Initialization and Startup Review

*   **Goal:** Review and refactor initialization logic in `resource_manager.py` and `main.py`.
*   **Files:**
    *   `app/core/resource_manager.py`
    *   `app/main.py`
*   **Subtasks:**
    *   [RELEVANT] [MODERATE] Analyze `ResourceManager.__init__` and the `startup` function in `main.py`. Identify potentially blocking operations (especially SSH connections or initial status checks if synchronous), complex setup logic, or background task initializations.
    *   [RELEVANT] [MODERATE] Refactor identified areas. Ensure potentially long-running initializations (like connecting to SLURM or fetching initial states) are handled asynchronously or appropriately deferred if possible. Ensure background tasks are started correctly within FastAPI's lifespan events.

## TASK 5: Error handling and Http Exceptions

*   **Goal:** Ensure `HTTPException`s are only raised from the `routes` layer, not the `core` layer. Implement custom exceptions in `core` for signaling specific error conditions.
*   **Files:**
    *   `app/core/resource_manager.py`
    *   `app/core/osire_llm_service.py`
    *   `app/routes/osireLLM.py`
    *   `app/routes/admin.py`
    *   `app/routes/model_proxy.py`
    *   `app/core/exceptions.py` (New file)
*   **Subtasks:**
    *   [RELEVANT] [MINOR] Create `app/core/exceptions.py`. Define base and specific custom exception classes (e.g., `OsireError`, `JobNotFound`, `InvalidStateError`, `ResourceManagerError`).
    *   [RELEVANT] [MAJOR] Search `app/core/*` files for `HTTPException`. Replace instances with appropriate custom exceptions defined in `app/core/exceptions.py`. Ensure core functions return values or raise these custom exceptions to signal outcomes.
    *   [RELEVANT] [MAJOR] Refactor exception handling in `app/routes/*` files. Add `try...except` blocks around calls to `core` layer functions. Catch specific custom exceptions and raise corresponding `HTTPException`s with appropriate status codes and details.

## TASK 6: FastAPI Conventions & Route Refactoring

*   **Goal:** Refactor `routes` to adhere strictly to FastAPI best practices, focusing on proper use of Pydantic models for request/response validation and separating business logic into the `core` layer.
*   **Files:**
    *   `app/routes/osireLLM.py`
    *   `app/routes/admin.py`
    *   `app/routes/model_proxy.py`
    *   `app/core/models.py`
    *   `app/core/osire_llm_service.py`
*   **Subtasks:**
    *   [RELEVANT] [MODERATE] Review all route functions in `app/routes/*`. Identify any logic that goes beyond request handling, validation, calling service functions, and response formatting.
    *   [RELEVANT] [MODERATE] Ensure all request bodies and query parameters are validated using Pydantic models. Define specific `RequestModel` and `ResponseModel` Pydantic classes in `app/core/models.py` where appropriate.
    *   [RELEVANT] [MAJOR] Move any identified business logic or complex data manipulation from `routes` functions into helper methods within `app/core/osire_llm_service.py` or potentially `app/core/resource_manager.py` if more appropriate.
    *   [RELEVANT] [MODERATE] Ensure route functions explicitly use Pydantic `ResponseModel`s in their signature (`response_model=...`) for automatic response validation and documentation generation.

## TASK 7: General Code Quality Refactoring

*   **Goal:** Document other identified areas violating good software patterns during the execution of prior tasks.
*   **Files:** TBD (Based on findings)
*   **Subtasks:**
    *   [RELEVANT] [MODERATE] During Tasks 1-6, identify specific instances of code duplication (violating DRY), overly long or complex methods, tight coupling, or other pattern violations. Log the findings in a new file, `documentation/refactoring.md`