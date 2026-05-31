# syntax=docker/dockerfile:1.7

# Variables used across multi-stages
ARG UBUNTU_IMAGE_TAG=24.04
ARG CUDA_IMAGE_TAG=13.0.0-cudnn-runtime-ubuntu24.04
ARG UV_VERSION=0.7.13
ARG PYTHON_VERSION=3.12

# Paths used across multi-stages
ARG UV_PROJECT_DIR=/app
ARG UV_PROJECT_ENVIRONMENT=/opt/venv
ARG UV_PYTHON_DIR=/python

# Device used to build the Docker image (cpu or gpu)
ARG DEVICE=cpu

# Stage 1: uv
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# Stage 2: Base image
FROM ubuntu:${UBUNTU_IMAGE_TAG} AS base-cpu
FROM nvidia/cuda:${CUDA_IMAGE_TAG} AS base-gpu
FROM base-${DEVICE} AS base

# Define host user and group IDs
ARG HOST_GID=4444
ARG HOST_UID=4444

# Define Python and uv settings
ARG PYTHON_VERSION
ARG UV_PROJECT_DIR
ARG UV_PROJECT_ENVIRONMENT
ARG UV_PYTHON_DIR

# Define Python and uv environment variables
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TERM=xterm-256color \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT} \
    UV_PYTHON=python${PYTHON_VERSION} \
    UV_PYTHON_INSTALL_DIR=${UV_PYTHON_DIR}

ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:/usr/local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -y \
        ca-certificates \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user and writable uv directories
RUN userdel -r ubuntu 2>/dev/null || true && \
    groupadd --force --gid ${HOST_GID} appuser && \
    useradd --create-home --uid ${HOST_UID} --gid ${HOST_GID} appuser && \
    mkdir -p ${UV_PROJECT_DIR} ${UV_PROJECT_ENVIRONMENT} ${UV_PYTHON_DIR} && \
    chown -R appuser:appuser ${UV_PROJECT_DIR} ${UV_PROJECT_ENVIRONMENT} ${UV_PYTHON_DIR}

# Set project directory and default user
WORKDIR ${UV_PROJECT_DIR}
USER appuser

# Stage 3: Python environment
FROM base AS python-env

# Copy uv and dependency metadata
COPY --from=uv /uv /usr/local/bin/uv
COPY --chown=appuser:appuser pyproject.toml uv.lock README.md LICENSE ./
COPY --chown=appuser:appuser flowers/__init__.py flowers/__init__.py

# Install Python dependencies without installing the project
ARG DEVICE
RUN uv sync --locked --no-cache --no-default-groups --group ${DEVICE} --no-dev --no-install-project

# Stage 4: runtime image
FROM python-env AS runtime

# Prevent uv from downloading isolated Python builds
ENV UV_PYTHON_DOWNLOADS=never

# Copy project and install it as a regular package
ARG DEVICE
COPY --chown=appuser:appuser . .
RUN uv sync --locked --no-cache --no-default-groups --group ${DEVICE} --no-dev --no-editable

# Stage 5: Notebook image
FROM python-env AS notebook

# Prevent uv from downloading isolated Python builds
ENV UV_PYTHON_DOWNLOADS=never

# Copy project and install it for notebooks
ARG DEVICE
COPY --chown=appuser:appuser . .
RUN uv sync --locked --no-cache --no-default-groups --group ${DEVICE} --group notebook

# Stage 6: Development image
FROM python-env AS dev

# Prevent uv from downloading isolated Python builds
ENV UV_PYTHON_DOWNLOADS=never

# Copy project and install it for development
ARG DEVICE
COPY --chown=appuser:appuser . .
RUN uv sync --locked --no-cache --no-default-groups --group ${DEVICE} --group dev
