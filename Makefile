# Shell to use
SHELL := /bin/bash

# Docker variables
PROJECT_NAME = flower-garden
DOCKER_REGISTRY_IMAGE = $(PROJECT_NAME)
DEVICE = cpu
GIT_SHA = $(shell git rev-parse --short HEAD)
HOST_GID = $(shell id -g)
HOST_UID = $(shell id -u)
DOCKER_IMAGE_DEV = $(DOCKER_REGISTRY_IMAGE):dev-cpu-$(GIT_SHA)
DOCKER_IMAGE_NOTEBOOK = $(DOCKER_REGISTRY_IMAGE):notebook-$(DEVICE)-$(GIT_SHA)
DOCKER_IMAGE_RUNTIME = $(DOCKER_REGISTRY_IMAGE):runtime-$(DEVICE)-$(GIT_SHA)
DOCKER_WORK_DIR = /app
LOCAL_WORK_DIR = $(PWD)

# Docker build arguments
DOCKER_BUILD_ARGS = --build-arg HOST_GID=$(HOST_GID) \
              		--build-arg HOST_UID=$(HOST_UID) \
              		--build-arg DEVICE=$(DEVICE)

DOCKER_BUILD_ARGS_DEV = --build-arg HOST_GID=$(HOST_GID) \
              			--build-arg HOST_UID=$(HOST_UID) \
              			--build-arg DEVICE=cpu

# Docker runtime arguments
DOCKER_RUN_ARGS = -it --rm -v "$(LOCAL_WORK_DIR):$(DOCKER_WORK_DIR)"

# GPU flag
IS_GPU_AVAILABLE = $(shell command -v nvidia-smi 2> /dev/null)
IS_GPU_FLAG_AVAILABLE = $(shell docker run --help | grep gpus)

ifeq ($(DEVICE),gpu)
	ifdef IS_GPU_AVAILABLE
		ifdef IS_GPU_FLAG_AVAILABLE
			DOCKER_GPU_COMMAND = --gpus all
		else
			DOCKER_GPU_COMMAND = --runtime nvidia
		endif
	endif
endif

# Default port
port = 8888

# Default target
.PHONY: help build-image-dev build-image-notebook build-image-runtime dev notebook runtime
.DEFAULT_GOAL := help

help:
	@echo "Usage: make [target]"
	@echo "Available targets:"
	@echo "  build-image-dev      - Build the CPU development Docker image."
	@echo "  build-image-notebook - Build the notebook Docker image. Override device with 'make build-image-notebook DEVICE=gpu'."
	@echo "  build-image-runtime  - Build the runtime Docker image. Override device with 'make build-image-runtime DEVICE=gpu'."
	@echo "  dev                  - Run the CPU development container interactively."
	@echo "  notebook             - Launch Jupyter Notebook. Define a custom port with 'make notebook port={my_port}'."
	@echo "  runtime              - Run the runtime container interactively."

build-image-dev:
	DOCKER_BUILDKIT=1 docker build $(DOCKER_BUILD_ARGS_DEV) -t $(DOCKER_IMAGE_DEV) --target dev .

build-image-notebook:
	DOCKER_BUILDKIT=1 docker build $(DOCKER_BUILD_ARGS) -t $(DOCKER_IMAGE_NOTEBOOK) --target notebook .

build-image-runtime:
	DOCKER_BUILDKIT=1 docker build $(DOCKER_BUILD_ARGS) -t $(DOCKER_IMAGE_RUNTIME) --target runtime .

dev: build-image-dev
	docker run $(DOCKER_RUN_ARGS) $(DOCKER_IMAGE_DEV) \
		/bin/bash

notebook: build-image-notebook
	docker run $(DOCKER_RUN_ARGS) $(DOCKER_GPU_COMMAND) -p $(port):8888 $(DOCKER_IMAGE_NOTEBOOK) \
		jupyter notebook --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.password='' --no-browser

runtime: build-image-runtime
	docker run $(DOCKER_RUN_ARGS) $(DOCKER_GPU_COMMAND) $(DOCKER_IMAGE_RUNTIME) \
		/bin/bash
