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
LIMIT ?= 10 
SF ?= 50

# --- Training Configuration ---
EPOCHS ?= 1000 
BATCH ?= 256

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
clean_local:
	rm -rf ./dataset/MAESTRO_Dataset/processed &&  mkdir -p ./dataset/MAESTRO_Dataset/processed
	rm -f preprocess.log
	rm -f training.log
	rm -rf training_output
	rm -rf generated_music.mid
	rm -rf reconstruction_verification.png
clean-remote-preprocess:
	@echo ">>> Cleaning REMOTE preprocessing directory and logs..."
	ssh $(REMOTE_USER)@$(REMOTE_HOST) ' \
		rm -rf $(REMOTE_BASE_PATH)/$(PROJECT_NAME)/dataset/MAESTRO_Dataset/processed && \
		mkdir -p $(REMOTE_BASE_PATH)/$(PROJECT_NAME)/dataset/MAESTRO_Dataset/processed && \
    		rm -f $(REMOTE_BASE_PATH)/$(PROJECT_NAME)/preprocess.log'
	@echo ">>> Remote preprocessing artifacts cleaned successfully."
clean-remote-training:
	@echo ">>> Cleaning REMOTE training directory and logs..."
	ssh $(REMOTE_USER)@$(REMOTE_HOST) ' \
		rm -rf $(REMOTE_BASE_PATH)/$(PROJECT_NAME)/training_output && \
		rm -f $(REMOTE_BASE_PATH)/$(PROJECT_NAME)/training.log'
	@echo ">>> Remote training artifacts cleaned successfully."

preprocess_local: clean_local
	time .venv/bin/python preprocess_maestro.py  --visualize --limit=$(LIMIT) --sf=$(SF) | tee preprocess.log 
	feh  dataset/MAESTRO_Dataset/reconstruction_verification.png
	# du -sh  ./dataset/MAESTRO_Dataset/processed/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1_segment_0.pt

preprocess: clean_local sync
	@echo ">>> Starting remote preprocessing job in tmux session '$(TMUX_SESSION_NAME)'..."
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "tmux new -d -s $(TMUX_SESSION_NAME) ' \
		cd $(REMOTE_BASE_PATH)/$(PROJECT_NAME) && pwd;\
		rm -rf $(REMOTE_PROCESSED_DIR) && mkdir -p $(REMOTE_PROCESSED_DIR);\
		rm preprocess.log;\
		time .venv/bin/python preprocess_maestro.py  --sf=$(SF) | tee preprocess.log '"
	
	@echo ">>> Job started successfully on remote server."
	@echo ">>> To view progress, run: make attach"
	make attach

train : 
	@echo ">>> Starting remote preprocessing job in tmux session '$(TMUX_SESSION_NAME)'..."
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "tmux new -d -s $(TMUX_SESSION_NAME) ' \
		cd $(REMOTE_BASE_PATH)/$(PROJECT_NAME) && pwd;\
		rm -rf training.log;\
		time .venv/bin/python main.py --epochs $(EPOCHS) --batch-size $(BATCH)| tee training.log '"
	@echo ">>> Job started successfully on remote server."
	@echo ">>> To view progress, run: make attach"
	make attach

full-run : 	preprocess train pull-results

pull-results:	
	@echo ">>> Syncing results from remote server using rsync..."
	# Create the local directory if it doesn't exist
	mkdir -p training_output
	# Use rsync to efficiently sync the training output directory
	rsync -avz --progress \
	    $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_BASE_PATH)/$(PROJECT_NAME)/training_output/ \
	    ./training_output/
	# Sync the log file as well
	rsync -avz --progress \
	    $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_BASE_PATH)/$(PROJECT_NAME)/training.log \
	    .
	@echo ">>> Results successfully synced to the 'training_output' directory."

attach:
	@echo ">>> Attaching to tmux session '$(TMUX_SESSION_NAME)' on $(REMOTE_HOST)..."
	ssh -t $(REMOTE_USER)@$(REMOTE_HOST) "tmux attach -t $(TMUX_SESSION_NAME)"

kill-session:
	@echo ">>> Killing tmux session '$(TMUX_SESSION_NAME)' on $(REMOTE_HOST)..."
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "tmux kill-session -t $(TMUX_SESSION_NAME)" || true
	@echo ">>> Session killed."


play:
	@echo "Generating and playing bars "
	.venv/bin/python generate.py --num-bars 18
	timidity generated_music.mid
