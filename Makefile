# ==============================================================================
# Makefile for Project Synchronization with rsync
# ==============================================================================

# --- (REQUIRED) User Configuration ---
# Fill in your remote server details here.
REMOTE_USER := r15i
REMOTE_HOST := 4nt0n.local
REMOTE_BASE_PATH := /home/r15i/Desktop/projects

# --- Project Configuration ---
# This gets the name of the current directory (e.g., "DeepLearnigMidi")
PROJECT_NAME := $(shell basename "$(CURDIR)")
REMOTE_DEST := $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_BASE_PATH)/$(PROJECT_NAME)
EXCLUDE_FILE := .rsync-exclude
# --- rsync Flags ---
# -a: archive mode (preserves permissions, ownership, timestamps)
# -v: verbose output
# -z: compress file data during the transfer
# -h: human-readable numbers
# --delete: delete files on the destination that don't exist on the source
# --progress: show progress during transfer
RSYNC_FLAGS := -avzh --delete --progress

.PHONY: help sync 

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help                 Show this help message."
	@echo "  sync                 Sync project files to remote, excluding dataset and outputs."
	@echo ""
	@echo "Configuration:"
	@echo "  - Edit REMOTE_USER, REMOTE_HOST, and REMOTE_BASE_PATH in the Makefile."

run:	#TODO: add tee
	time python main.py >> out.log

run-no-logs:
	time python main.py 

sync:
	@echo ">>> Synchronizing project to $(REMOTE_DEST)"
	@echo ">>> Excluding files listed in $(EXCLUDE_FILE)"
	rsync $(RSYNC_FLAGS) --exclude-from=$(EXCLUDE_FILE) ./ $(REMOTE_DEST)
	@echo ">>> Synchronization complete."

