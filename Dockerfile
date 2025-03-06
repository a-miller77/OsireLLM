FROM python:3.11-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    build-essential \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --ignore-installed pipenv

WORKDIR /var/task

COPY Pipfile /var/task/
RUN if [ -f Pipfile.lock ]; then cp Pipfile.lock /var/task/; fi

COPY app /var/task/app

RUN if ! [ -f /var/task/Pipfile.lock ]; then pipenv lock; fi

RUN pipenv requirements > requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

# Add development tools
RUN pip install pytest debugpy ipdb

# Create a volume mount point for development
VOLUME ["/var/task/app"]

# Use entrypoint for more flexibility
ENTRYPOINT []
CMD ["uvicorn", "--app-dir", "app", "main:app", "--port", "8080"]