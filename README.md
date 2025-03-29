# OsireLLM

This project serves as the central orchestrator for launching, managing, and interacting with Large Language Models (LLMs) deployed as vLLM instances on the Rosie HPC cluster via SLURM. It also provides access to any configured globally accessable models. It provides a unified interface for users to manage model lifecycles and proxy requests to the appropriate running model server.

**Status:** Active Development

## Key Features

*   **SLURM Job Management:** Launch, monitor, and terminate vLLM server instances as SLURM jobs on Rosie partitions (including GPU allocation).
*   **Model Proxying:** Route incoming OpenAI-compatible API requests (`/v1/...`) to the correct running NIM or vLLM backend based on the specified `model` parameter. Supports streaming responses.
*   **Dynamic API Documentation:** Automatically fetches the OpenAPI schema from running NIM and vLLM instances and merges relevant endpoints (`/v1/...`) into this gateway's `/docs` page, providing up-to-date documentation for available models.
*   **Resource Management:** Manages port allocation for vLLM instances and tracks job states for all models. Includes background tasks for monitoring and cleaning up finished/failed jobs.
*   **Static Model Support:** Can integrate with statically deployed models (e.g., a dedicated DGX server instance).
*   **Async Operations:** Leverages asynchronous programming (`asyncio`, `asyncssh`, `httpx`) for efficient handling of API requests and background tasks.
*   **Secure Cluster Interaction:** Uses SSH keys (`asyncssh`, Fabric) for secure communication with SLURM management nodes to submit and control jobs. Includes automated SSH key setup for streamlined local interaction.
*   **Admin Controls:** Provides endpoints for health checks, system statistics, and dynamic log level adjustments.

## Architecture Overview

OsireLLM is a FastAPI application that acts as a control plane and proxy:

1.  **User Interaction:** Users interact with the Osire Ingress API endpoints (e.g., `/launch`, `/status`, `/v1/chat/completions`).
2.  **Orchestration (`OsireLLM` routes):** Launch requests trigger the `resource_manager`.
3.  **SLURM Interaction (`resource_manager`, `shell_commands`):** The `resource_manager` uses `asyncssh` (preferred) or Fabric to connect to Rosie's management nodes (`dh-mgmt*`). It generates and submits SLURM batch scripts (`sbatch`), checks job status (`scontrol`, `sacct`), and cancels jobs (`scancel`). It also manages port allocation and job state tracking. Requires passwordless SSH key access from where this service runs *to* the management nodes.
4.  **Proxying (`model_proxy` routes):** API calls to model endpoints (e.g., `/v1/...`) are intercepted. The required model is identified, its running location (host/port) is retrieved from the `resource_manager`, and the request is forwarded using `httpx`. Streaming is supported.
5.  **Dynamic Docs (`osire_llm_service`):** When a NIM server is detected at startup (or manually triggered), its OpenAPI schema is fetched. Relevant paths are extracted and merged into the OsireLLMs FastAPI application's own schema, making them visible in `/docs`.
6.  **Background Tasks:** Periodic tasks update job statuses and clean up resources associated with completed or failed jobs.

## Roadmap

This outlines the planned future developments for OsireLLM

1.  **Core Refactoring & Integration:**
    *   Reduce technical debt and refactor code using improved software patterns for better maintainability and extensibility.
    *   Integrate with the wider Osire platform:
        *   Replace hardcoded paths (e.g., output directories, container locations) with values from Osire's configuration system.
        *   Integrate with the Osire authentication layer.
        *   Register OsireLLM with the Osire Service Registry.
    *   Enhance the `JobStatus` model to store engine-specific information (e.g., differentiation between vLLM and NIM).
2.  **Advanced Endpoint Proxying:**
    *   Implement flexible proxy endpoints (`/<engine>/<model>/path:path`) to allow direct access to the full API of the underlying LLM engine servers.
3.  **Expanded Engine Support:**
    *   Add full lifecycle management support for NVIDIA NIM endpoints (beyond just proxying existing ones). This may require a dedicated NIM Osire Service for HPC that doesn't have docker support.
    *   Add support for launching and managing SGLang instances as a backend engine.
4.  **Further Enhancements:**
    *   Develop more sophisticated monitoring and logging capabilities.
    *   Introduce more granular resource management options during job launch (e.g., specific GPU types, memory constraints).
