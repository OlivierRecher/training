# Makefile for the training project -- data preparation pipeline
# Quick usage:
#   make data-train    -> build the timestamp-aware N_TRAIN/N_TEST sample dataset
#                         (override counts/source with N_TRAIN=.../N_TEST=.../SRC=...)
#   make clean         -> remove generated dataset files and Python caches
#   make help          -> show help
#
# Requires the virtual environment created by `bash setup_venv.sh` -- run that
# first if `.venv` does not exist yet.

VENV_PYTHON := .venv/bin/python
PYTHON_DIR  := python
DATA_DIR    := data/dataset_for_training

# Number of train/test users to sample for the `data-train` target
# (override: make data-train N_TRAIN=1000 N_TEST=200)
N_TRAIN := 400
N_TEST  := 100
# Sample day-file to split (unfiltered) for the `data-train` target
# (override: make data-train SRC=data/sample_for_training/2014-03-13_sample_for_training.csv)
SRC := data/sample_for_training/2014-03-12_sample_for_training.csv

.DEFAULT_GOAL := help

$(VENV_PYTHON):
	@echo "ERREUR: .venv introuvable -- lancez d'abord: bash setup_venv.sh" >&2
	@exit 1

# --- Sampled training dataset (N_TRAIN + N_TEST users), timestamps kept -----
# Splits the sample_for_training day-file into an EXACT train/test count with
# NO filtering at all -- userId/nRecord/cellId/timestamp cleaning (timestamps
# KEPT this time, turned into an hour-of-day signal by format_for_train.py), a
# random pick of N_TRAIN + N_TEST users stratified by nRecord, then JSONL
# formatting.
#   make data-train                          -> 400 train + 100 test (default)
#   make data-train N_TRAIN=1000 N_TEST=200  -> 1000 train + 200 test
#   make data-train SRC=path/to/other.csv    -> split a different day-file
#   -> data/dataset_for_training/{N_TRAIN}_users_train.jsonl
#   -> data/dataset_for_training/{N_TRAIN}_users_test.jsonl
.PHONY: data-train
data-train: $(VENV_PYTHON)  ## Build the train/test sample dataset + format; N_TRAIN=400/N_TEST=100 by default
	@echo ">> Splitting $(N_TRAIN) train + $(N_TEST) test users (no filters) from $(SRC)"
	$(VENV_PYTHON) $(PYTHON_DIR)/split_sample_for_training.py $(SRC) --n-train $(N_TRAIN) --n-test $(N_TEST)
	$(VENV_PYTHON) $(PYTHON_DIR)/format_for_train.py $(DATA_DIR)/$(N_TRAIN)_users.csv --n-train $(N_TRAIN) --n-test $(N_TEST)

# --- Groupes de comportement (cross-training) -------------------------------
# Rapport sur les groupes utilisés par le cross-training du notebook : profil de
# chaque groupe, créneaux de transition dérivés, R² de prédictibilité expliquée,
# détectabilité depuis un préfixe, et placement des créneaux vs créneau global.
#   make groups                       -> rapport sur le jeu 400/100 par défaut
#   make groups N_TRAIN=1000          -> rapport sur {N_TRAIN}_users_{train,test}.jsonl
#   make groups GROUPS=6              -> essayer un autre nombre de groupes
GROUPS := 4
.PHONY: groups
groups: $(VENV_PYTHON)  ## Analyse des groupes de comportement (GROUPS=4 par défaut)
	$(VENV_PYTHON) $(PYTHON_DIR)/analyze_user_groups.py \
		--train $(DATA_DIR)/$(N_TRAIN)_users_train.jsonl \
		--test $(DATA_DIR)/$(N_TRAIN)_users_test.jsonl \
		--n-groups $(GROUPS)

# --- Autonomie du notebook ---------------------------------------------------
# Le notebook embarque une copie de python/*.py pour tourner sur une machine
# vierge sans fichier annexe. À relancer après toute modification de ces modules
# (`make test` échoue sinon, pour éviter une copie embarquée obsolète).
.PHONY: sync-notebook
sync-notebook: $(VENV_PYTHON)  ## Réembarque python/*.py dans le notebook (autonomie)
	$(VENV_PYTHON) $(PYTHON_DIR)/sync_notebook_fallbacks.py

.PHONY: test
test: $(VENV_PYTHON)  ## Run the test suite
	$(VENV_PYTHON) -m pytest -q

.PHONY: clean
clean:              ## Remove generated dataset files and Python caches
	@echo ">> Removing generated datasets and caches"
	rm -rf $(DATA_DIR)
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +

.PHONY: help
help:               ## Show this help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
