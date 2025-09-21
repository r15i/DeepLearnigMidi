# ==============================================================================
# Makefile for Project Synchronization and Remote Execution
# ==============================================================================

# --- (REQUIRED) User Configuration ---
REMOTE_USER := r15i
REMOTE_HOST := 4nt0n.local
REMOTE_BASE_PATH := /home/r15i/Desktop/projects

# --- Project Configuration ---
PROJECT_NAME := $(shell basename "$(CURDIR)")
REMOTE_DEST := $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_BASE_PATH)/$(PROJECT_NAME)
EXCLUDE_FILE := .rsync-exclude
REMOTE_PROCESSED_DIR := $(REMOTE_BASE_PATH)/$(PROJECT_NAME)/dataset/MAESTRO_Dataset/processed

# --- NEW: Virtual Environment and Pip Command ---
# Assumes a .venv directory in your project root
VENV_PYTHON := .venv/bin/python
PIP_CMD := uv pip

# --- tmux Configuration ---
TMUX_SESSION_NAME := preprocessing

# --- rsync Flags ---
RSYNC_FLAGS := -avzh --delete --progress

# --- Preprocessing Configuration ---
LIMIT ?= None
SF ?= 100

.PHONY: help setup sync preprocess attach kill-session

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help          Show this help message."
	@echo "  setup         Create virtual env and install dependencies from requirements.txt."
	@echo "  sync          Sync project files to the remote server."
	@echo "  preprocess    Sync, setup remote env, and run preprocessing in a tmux session."
	@echo "                Pass arguments like so: make preprocess LIMIT=100 SF=25"
	@echo "  attach        Attach to the running tmux session on the remote server."
	@echo "  kill-session  Kill the remote tmux session."

setup:
	@echo ">>> Setting up Python virtual environment..."
	python3 -m venv .venv
	@echo ">>> Installing dependencies from requirements.txt..."
	$(VENV_PYTHON) -m $(PIP_CMD) install -r requirements.txt
	@echo ">>> Setup complete."

sync:
	@echo ">>> Synchronizing project to $(REMOTE_DEST)"
	rsync $(RSYNC_FLAGS) --exclude-from=$(EXCLUDE_FILE) ./ $(REMOTE_DEST)
	@echo ">>> Synchronization complete."

preprocess: sync
	@echo ">>> Starting remote preprocessing job in tmux session '$(TMUX_SESSION_NAME)'..."
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "tmux new -d -s $(TMUX_SESSION_NAME) ' \
		cd $(REMOTE_BASE_PATH)/$(PROJECT_NAME) && pwd;\
		rm -rf $(REMOTE_PROCESSED_DIR) && mkdir -p $(REMOTE_PROCESSED_DIR);\
		rm preprocess.log;\
		time .venv/bin/python preprocess_maestro.py  --sf=$(SF) | tee preprocess.log '"
	
	@echo ">>> Job started successfully on remote server."
	@echo ">>> To view progress, run: make attach"

attach:
	@echo ">>> Attaching to tmux session '$(TMUX_SESSION_NAME)' on $(REMOTE_HOST)..."
	ssh -t $(REMOTE_USER)@$(REMOTE_HOST) "tmux attach -t $(TMUX_SESSION_NAME)"

kill-session:
	@echo ">>> Killing tmux session '$(TMUX_SESSION_NAME)' on $(REMOTE_HOST)..."
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "tmux kill-session -t $(TMUX_SESSION_NAME)" || true
	@echo ">>> Session killed."
