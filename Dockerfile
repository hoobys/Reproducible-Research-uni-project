# Use the official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Update the package list and install the necessary packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Set the working directory in the Docker image
WORKDIR /app

# Copy only the pyproject.toml and optionally the poetry.lock file first to cache dependencies
# COPY ./pyproject.toml ./

COPY . .
# Copy the rest of our application code into the working directory
# COPY ./raytrans_rag /raytrans_rag

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

