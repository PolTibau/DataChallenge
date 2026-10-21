# Enedis Load Curve Imputation

Imputation of missing values in half-hourly electricity consumption curves, developed for
the ENS Data Challenge proposed by Enedis. Each curve covers 1057 half-hours (about
22 days) and contains gaps of varying length; the goal is to reconstruct the missing
points. Two approaches are implemented and compared:

| Approach | Idea |
|---|---|
| **KNN imputer** (`knn/`) | fill each gap with a weighted average of the most similar reference curves |
| **Transformer masked autoencoder** (`transformer/`) | learn to reconstruct artificially masked segments, using rich temporal and cross-consumer context |

**Metric:** mean absolute error in watts, computed only on the points that were originally missing.

## Methods

### KNN imputer

For each incomplete curve, the K = 8 closest reference curves are found using the
Manhattan distance over the timesteps observed in both curves (with a minimum overlap).
Gaps are filled with a rank-weighted average of the neighbours. Hyperparameters
(K, distance, weighting, normalisation) are tuned with Bayesian optimisation (Optuna).

### Transformer masked autoencoder

An encoder–decoder Transformer (d_model 256, 8 heads, 4 + 4 layers, one token per hour)
trained on complete reference curves in which gaps are simulated, following the empirical
distribution of gap lengths observed in the data.

Each half-hour is described by 16 input channels (see `transformer/core/context.py`):

- the normalised value, the missing-value mask and a linear interpolation;
- the same hour one day and one week before and after;
- the consumer's usual profile for that time slot (working day or holiday);
- the population curve and its anomaly (a proxy for weather effects);
- the average of the 50 most similar consumers and its anomaly.

A per-curve descriptor (10 consumption indicators, including temperature sensitivity) is
added as an extra embedding. The loss is weighted by the scale of each curve, so that it
matches the MAE in watts used for scoring.

At inference time, the best checkpoints of every training run are averaged, and for very
short gaps (1–2 points) the prediction is blended with linear interpolation using weights
tuned on the labelled validation curves.

## Requirements

- Python 3 with the packages listed in `requirements.txt` (numpy, pandas, matplotlib,
  tqdm, scikit-learn, torch, optuna).
- A CUDA GPU is recommended for training the Transformer.
- Input data (not included because of its size): download `X_train.csv`, `y_train.csv` and
  `X_test.csv` from the challenge page and place them in `data/` at the repository root.

## Installation

```bash
# PyTorch with CUDA (NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Remaining dependencies
pip install -r requirements.txt
```

## Usage

Run every command from the repository root.

### KNN

```bash
python -m knn.optimize_hyperparams   # optional: Bayesian hyperparameter search (Optuna)
python -m knn.main_train             # evaluate on the training set and save predictions
python -m knn.main_predict           # generate test predictions -> outputs/knn/
```

### Transformer

```bash
# Quick check with few curves and epochs
TRANSFORMER_DEBUG=1 python -m transformer.main_train

# Training: each run is saved under transformer/checkpoints/seed<N>/
python -m transformer.main_train                    # default seed (42)
python -m transformer.main_train --seed 43          # additional runs for the ensemble
python -m transformer.main_train --seed 42 --resume # resume an interrupted run

# Prediction: combines all runs -> outputs/transformer/y_pred_test.csv
python -m transformer.main_predict
python -m transformer.main_predict --runs seed42 seed43   # only selected runs
```

Settings that can be changed through environment variables (see
`transformer/core/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `TRANSFORMER_PATCH_SIZE` | 2 | half-hours per token (1 = finer, about 4× slower) |
| `TRANSFORMER_BATCH_SIZE` | 64 | batch size |
| `TRANSFORMER_WORKERS` | 4 | data-loading processes (use 0 on Windows) |
| `TRANSFORMER_DEBUG` | 0 | 1 = quick run on a small subset |

`notebooks/kaggle_training.ipynb` runs training and prediction on a Kaggle GPU.

## Project structure

```
.
├── data/                      # input CSVs (not versioned)
├── outputs/                   # predictions (not versioned)
├── src/
│   └── data_loader.py         # data loading shared by both approaches
├── knn/
│   ├── core/                  # configuration, preprocessing, KNN imputer
│   ├── main_train.py
│   ├── main_predict.py
│   └── optimize_hyperparams.py
├── transformer/
│   ├── core/
│   │   ├── config.py          # paths, architecture and training settings
│   │   ├── context.py         # the 16 context channels
│   │   ├── dataset.py         # consumer descriptors, simulated gaps, datasets
│   │   ├── model.py           # Transformer masked autoencoder
│   │   ├── pipeline.py        # preprocessing and post-processing shared by train/predict
│   │   └── trainer.py         # training loop, checkpoints, ensemble inference
│   ├── checkpoints/           # trained weights (not versioned)
│   ├── main_train.py
│   ├── main_predict.py
│   └── test_resume_smoke.py
├── notebooks/
│   ├── exploration.ipynb      # exploratory data analysis
│   └── kaggle_training.ipynb  # training on Kaggle
└── requirements.txt
```

## Tests

`transformer/test_resume_smoke.py` runs the whole Transformer pipeline on synthetic data:
a short training, resuming from a checkpoint, and ensemble prediction.

```bash
TRANSFORMER_WORKERS=0 python -m transformer.test_resume_smoke
```

## License

Released under the [MIT License](LICENSE). The challenge data belongs to its owners and is
not covered by this license.
