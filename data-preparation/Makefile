# Makefile for the Stage_2A project
# Quick usage:
#   make               -> create the venv + install dependencies (the "install" target)
#   make data-train    -> build the 400/100 train/test sample dataset for training
#                         (override counts/source with N_TRAIN=.../N_TEST=.../SRC=...)
#   make data-n-user   -> build the 1900/900 train + 100 test JSONL sets from
#                         2000_users.csv, and convert the padded 400/100 files
#   make clean         -> remove the venv and Python caches
#   make help          -> show help

# --- Configuration ---------------------------------------------------------
VENV        := venv
PYTHON      := python3
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP    := $(VENV)/bin/pip
REQS        := requirements.txt
# Sentinel file: marks that dependencies have been installed
STAMP       := $(VENV)/.installed
# Number of train/test users to sample for the `data-train` target
# (override: make data-train N_TRAIN=1000 N_TEST=200)
N_TRAIN     := 400
N_TEST      := 100
# Sample day-file to split (unfiltered) for the `data-train` target
# (override: make data-train SRC=Database/sample_for_training/2014-03-13_sample_for_training.csv)
SRC := Database/sample_for_training/2014-03-12_sample_for_training.csv

.DEFAULT_GOAL := install

# --- Virtualenv creation ---------------------------------------------------
$(VENV_PYTHON):
	@echo ">> Creating the virtual environment in ./$(VENV)"
	$(PYTHON) -m venv $(VENV)
	@$(VENV_PIP) install --upgrade pip

# --- Dependency installation ----------------------------------------------
# Re-runs automatically if requirements.txt changes.
$(STAMP): $(VENV_PYTHON) $(REQS)
	@echo ">> Installing dependencies from $(REQS)"
	$(VENV_PIP) install -r $(REQS)
	@touch $(STAMP)

.PHONY: install
install: $(STAMP)   ## Create the venv and install all dependencies
	@echo ""
	@echo "============================================================"
	@echo " Environment ready!"
	@echo "------------------------------------------------------------"
	@echo " Next steps:"
	@echo "   1) Activate the venv:  source $(VENV)/bin/activate"
	@echo "   2) Run your scripts from the Python/ directory."
	@echo "============================================================"

# --- Sampled training dataset (N_TRAIN + N_TEST users) -----------------------
# Splits the sample_for_training day-file into an EXACT train/test count with
# NO filtering at all — just userId/nRecord/cellId cleaning (timestamps
# dropped) followed by a random pick of N_TRAIN + N_TEST users, then Unsloth
# formatting.
#   make data-train                          -> 400 train + 100 test (default)
#   make data-train N_TRAIN=1000 N_TEST=200  -> 1000 train + 200 test
#   make data-train SRC=path/to/other.csv    -> split a different day-file
#   -> Database/dataset_for_training/{N_TRAIN}_{N_TEST}_sample_users_train.jsonl
#   -> Database/dataset_for_training/{N_TRAIN}_{N_TEST}_sample_users_test.jsonl
.PHONY: data-train
data-train: $(STAMP)  ## Build the train/test sample dataset + format; N_TRAIN=400/N_TEST=100 by default
	@echo ">> Splitting $(N_TRAIN) train + $(N_TEST) test users (no filters) from $(SRC)"
	cd Python && ../$(VENV_PYTHON) split_sample_for_training.py ../$(SRC) --n-train $(N_TRAIN) --n-test $(N_TEST)
	cd Python && ../$(VENV_PYTHON) format_for_train.py ../Database/dataset_for_training/$(N_TRAIN)_$(N_TEST)_sample_users.csv --n-train $(N_TRAIN) --n-test $(N_TEST)

# --- N-user JSONL datasets (data-n-user) -----------------------------------
# Builds the JSONL training sets used to study the effect of training-set size,
# from two kinds of source:
#
#   1) USERS_SRC — a RAW vektory day-file (no header). Cleaned (timestamps
#      dropped), then split stratified-by-nRecord into N_USERS_TEST held-out
#      test users + N_USERS_TRAIN train users, plus a stratified
#      N_USERS_SUBSET-user subset of that train set (drawn from the train rows
#      only, so no test user ever leaks into a train file).
#   2) PADDED_TRAIN / PADDED_TEST — already cleaned, padded and split, so they
#      are converted as-is (--all), one JSONL per file.
#
# Everything is written to OUT_DIR (= PADDED_DIR, i.e. next to the sources).
#
#   make data-n-user
#   make data-n-user N_USERS_TRAIN=1000 N_USERS_SUBSET=500
#   make data-n-user OUT_DIR=Database/dataset_for_training
#   -> Database/2000_1000_500padded/2000_users_1900_train.jsonl
#   -> Database/2000_1000_500padded/2000_users_900_train.jsonl
#   -> Database/2000_1000_500padded/2000_users_100_test.jsonl
#   -> Database/2000_1000_500padded/400_100_sample_users_train_padded.jsonl
#   -> Database/2000_1000_500padded/400_100_sample_users_test_padded.jsonl
PADDED_DIR      := Database/2000_1000_500padded
# Inputs live in PADDED_DIR/source/, generated files go to PADDED_DIR itself
# (override either with SRC_DIR=... / OUT_DIR=...)
SRC_DIR         := $(PADDED_DIR)/source
OUT_DIR         := $(PADDED_DIR)
USERS_SRC       := $(SRC_DIR)/2000_users.csv
PADDED_TRAIN    := $(SRC_DIR)/400_100_sample_users_train_padded.csv
PADDED_TEST     := $(SRC_DIR)/400_100_sample_users_test_padded.csv
N_USERS_TRAIN   := 1900
N_USERS_TEST    := 100
N_USERS_SUBSET  := 900
USERS_STEM      := $(basename $(notdir $(USERS_SRC)))

.PHONY: data-n-user
data-n-user: $(STAMP)  ## Build the 1900/900 train + 100 test JSONL sets and convert the padded 400/100 files
	@echo ">> Splitting $(N_USERS_TRAIN) train + $(N_USERS_TEST) test (+ $(N_USERS_SUBSET)-user train subset) from $(USERS_SRC)"
	cd Python && ../$(VENV_PYTHON) split_sample_for_training.py ../$(USERS_SRC) \
		--out ../$(OUT_DIR) --n-train $(N_USERS_TRAIN) --n-test $(N_USERS_TEST) \
		--train-subset $(N_USERS_SUBSET)
	@echo ">> Formatting the split CSVs to JSONL"
	cd Python && ../$(VENV_PYTHON) format_for_train.py ../$(OUT_DIR)/$(USERS_STEM)_$(N_USERS_TRAIN)_train.csv --out ../$(OUT_DIR) --all
	cd Python && ../$(VENV_PYTHON) format_for_train.py ../$(OUT_DIR)/$(USERS_STEM)_$(N_USERS_SUBSET)_train.csv --out ../$(OUT_DIR) --all
	cd Python && ../$(VENV_PYTHON) format_for_train.py ../$(OUT_DIR)/$(USERS_STEM)_$(N_USERS_TEST)_test.csv --out ../$(OUT_DIR) --all
	@echo ">> Formatting the pre-padded 400/100 files to JSONL (converted as-is)"
	cd Python && ../$(VENV_PYTHON) format_for_train.py ../$(PADDED_TRAIN) --out ../$(OUT_DIR) --all
	cd Python && ../$(VENV_PYTHON) format_for_train.py ../$(PADDED_TEST) --out ../$(OUT_DIR) --all

# --- Cleanup ---------------------------------------------------------------
.PHONY: clean
clean:              ## Remove the venv and Python caches
	@echo ">> Removing the venv and caches"
	rm -rf $(VENV)
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +

# --- Help ------------------------------------------------------------------
.PHONY: help
help:               ## Show this help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
