# Stage 1: Builder/Development Environment
FROM python:3.11-slim-bullseye AS builder

# Install OS dependencies and pipenv
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    build-essential \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --ignore-installed pipenv

WORKDIR /var/task

# Copy dependency definition files first for caching
COPY Pipfile Pipfile.lock* /var/task/
# Try to lock if lock file missing (less preferred than committed lock)
RUN if ! [ -f /var/task/Pipfile.lock ]; then echo "Warning: Pipfile.lock not found, generating..." && pipenv lock; fi

# Install ALL packages (including dev) from Pipfile.lock into the system Python
# Use --deploy for stricter lock file adherence
RUN pipenv install --system --dev --deploy --ignore-pipfile

# Copy application code
COPY app /var/task/app
# Copy tests (needed if running tests in this stage)
COPY tests /var/task/tests
COPY pytest.ini /var/task/


# Stage 2: Final/User Runtime Environment
FROM python:3.11-slim-bullseye AS final

WORKDIR /var/task

# Copy only installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# Ensure python can find packages
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages

# Copy application code from the builder stage
COPY --from=builder /var/task/app /var/task/app

# Application entrypoint/cmd (No dev tools like --reload needed here)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Note: Removed VOLUME instruction, typically not needed in final image
# Note: ENTRYPOINT removed for simplicity, CMD is usually sufficient