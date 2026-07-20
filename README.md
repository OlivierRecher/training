# Prédiction du prochain `cell_id` — fine-tuning LoRA d'un LLM

Solution d'entraînement d'un LLM avec **LoRA** qui prédit les prochains `cell_id` d'une séquence
d'événements utilisateur, à partir de `100_users_train.jsonl`. Deux modèles de base au choix dans
la cellule CONFIG : **Mistral-7B-v0.1** (GPU Colab obligatoire, chargé en 4-bit/QLoRA, licence à
accepter sur Hugging Face) ou **Qwen2.5-0.5B-Instruct** (léger, fonctionne aussi en local sur MPS).

**Objectif : accuracy top-1 ≥ 80 %** sur `100_users_test.jsonl` (20 utilisateurs jamais vus).
Baseline « répéter le cell_id précédent » : 70,8 %.

## Contenu

| Fichier | Rôle |
|---|---|
| `train_cellid_llm.ipynb` | Notebook unique (local **et** Google Colab) : données → modèle → entraînement avec guards → évaluation → inférence |
| `setup_venv.sh` | Crée le venv Python 3.11, installe les dépendances, enregistre le kernel Jupyter |
| `requirements.txt` | Dépendances (torch, transformers, peft, …) |
| `100_users_train.jsonl` | 80 séquences d'entraînement (72 train / 8 validation) |
| `100_users_test.jsonl` | 20 séquences pour l'évaluation finale uniquement |

Sorties générées dans `outputs/` : `checkpoints/` (2 max), `best_model/` (meilleure accuracy
validation), `final_model/` (adaptateur LoRA + tokenizer en fin d'entraînement).

## Utilisation en local (Mac/Linux)

```bash
bash setup_venv.sh                      # une seule fois
source .venv/bin/activate && jupyter lab
```

Ouvrir `train_cellid_llm.ipynb` avec le kernel **Python (cellid-llm)** et exécuter toutes les
cellules. Sur un Mac M3 (MPS), l'entraînement complet prend quelques dizaines de minutes.

## Utilisation sur Google Colab

1. Ouvrir [colab.research.google.com](https://colab.research.google.com) → importer `train_cellid_llm.ipynb`
2. Uploader `100_users_train.jsonl` et `100_users_test.jsonl` via le panneau **Fichiers** (icône dossier à gauche)
3. Runtime → *Modifier le type d'exécution* → **GPU (T4)**
4. Exécuter toutes les cellules — l'environnement Colab est détecté automatiquement (installation
   des dépendances + device CUDA), l'entraînement prend ~2-5 min

## Garde-fous (protection de la machine locale)

- **Préflight** : vérifie RAM libre (≥ 2 Go), espace disque (≥ 5 Go) et présence des données avant de lancer
- **Guard RAM** : la mémoire est surveillée pendant l'entraînement ; si elle devient critique
  (> 90 % utilisée ou < 2 Go libres), un checkpoint est sauvegardé puis l'entraînement s'arrête
  proprement — pas de gel ni de swap massif
- **Checkpoints bornés** : au plus 2 checkpoints conservés sur disque, reprise possible avec
  `CONFIG["resume_from_checkpoint"] = True`
- **CPU préservé** : threads PyTorch limités à la moitié des cœurs en local
- **Arrêt automatique** : l'entraînement s'arrête dès que la cible de validation est atteinte
  (82 %), en cas de plateau (8 epochs sans progrès) ou au plafond de 60 epochs

## Approche technique

1. Les 264 `cell_id` sont ajoutés au tokenizer comme **tokens dédiés** (1 cell_id = 1 token) —
   la prédiction du prochain cell_id devient un simple argmax sur ces 264 tokens
2. Fine-tuning **LoRA** (r=16, α=32) sur les projections attention + MLP, plus
   `trainable_token_indices` pour entraîner uniquement les embeddings des nouveaux tokens
   (~9 M de paramètres entraînables sur 500 M, poids liés embed/lm_head gérés)
3. Loss causale calculée **uniquement sur les tokens cell_id** (le préfixe est masqué)
4. Évaluation en teacher forcing, identique aux baselines : pour chaque position i ≥ 1,
   prédire le cell i à partir du contexte 0..i-1 (top-1 / top-3 / top-5)

## Réglages si l'objectif n'est pas atteint

Dans la cellule CONFIG : augmenter `max_epochs`, passer `lora_r` à 32 et `lora_alpha` à 64,
ou ajuster `learning_rate`. Noter que ~4 % des cibles du test sont des `cell_id` absents du
train : le plafond théorique est ≈ 96 %.
