# Enedis Load Curve Imputation

Two approaches to the Enedis data challenge on imputing missing values (gaps) in
electricity consumption curves: a KNN imputer and a Transformer-based Temporal
Masked Autoencoder.

## Features

- **KNN imputer** (`knn/`): finds the K most similar reference curves (Manhattan
  distance over valid timesteps) and fills gaps with a rank-weighted average.
  Hyperparameter search with Optuna.
- **Transformer / Temporal Masked Autoencoder** (`lstm/`): trained on reference
  curves with simulated gaps. Each half-hour receives the value plus 16 context
  channels (mask, interpolation, same hour ±1 day/week, usual consumer profile,
  population curve, 50 most similar consumers; see `lstm/src/context.py`). The
  best checkpoints of all runs are averaged and blended with interpolation for
  gaps of 1-2 points.
- **Metric**: MAE in watts, computed only on points that were originally NaN.

> The Transformer package is named `lstm` for historical reasons; it does not use an LSTM.

## Requirements

- Python 3 with the packages in `requirements.txt` (numpy, pandas, matplotlib,
  tqdm, scikit-learn, torch, optuna).
- A CUDA GPU is recommended for Transformer training.
- Input data (not included, too large): download `X_train.csv`, `y_train.csv` and
  `X_test.csv` from the Enedis data challenge page and place them in `data/` at
  the repository root.

## Installation

```bash
# PyTorch with CUDA (NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Remaining dependencies
pip install -r requirements.txt
```

## Usage

Run everything from the repository root.

### KNN

```bash
python -m knn.optimize_hyperparams   # optional Bayesian optimisation (Optuna)
python -m knn.main_train             # evaluate on train and save predictions
python -m knn.main_predict           # generate test predictions
```

### Transformer

```bash
# Quick check (few epochs, few curves)
LSTM_DEBUG=1 python -m lstm.main_train_combined

# Train (saved under lstm/checkpoints/v4/seed42/)
python -m lstm.main_train_combined
python -m lstm.main_train_combined --seed 43       # extra runs for the ensemble
python -m lstm.main_train_combined --seed 42 --resume

# One token per half hour (slower)
LSTM_PATCH_SIZE=1 LSTM_BATCH_SIZE=32 python -m lstm.main_train_combined --seed 44

# Prediction: combines all seed*/ runs -> outputs/lstm/y_pred_test_v4.csv
python -m lstm.main_predict
python -m lstm.main_predict --runs seed42 seed43   # only these runs
```

Environment variables read by `lstm/src/config.py`: `LSTM_PATCH_SIZE`,
`LSTM_BATCH_SIZE`, `LSTM_WORKERS`, `LSTM_DEBUG`.

`kaggle_train_combined.ipynb` runs the training on Kaggle.

## Project structure

```
.
├── data/                 # input CSVs (not versioned)
├── outputs/              # predictions (not versioned)
├── notebooks/            # exploratory analysis
├── src/                  # shared code: data loading
├── knn/
│   ├── core/             # config, preprocessing, KNN algorithm
│   ├── main_train.py
│   ├── main_predict.py
│   └── optimize_hyperparams.py
├── lstm/
│   ├── src/              # config, context, dataset, model, pipeline, trainer
│   ├── checkpoints/      # trained weights (not versioned)
│   ├── main_train_combined.py
│   ├── main_predict.py
│   └── test_resume_smoke.py
└── requirements.txt
```

## Tests

`lstm/test_resume_smoke.py` is a smoke test on synthetic data (short training,
resume, ensemble):

```bash
python -m lstm.test_resume_smoke
```

## Contributing

Issues and pull requests are welcome. Keep changes focused and describe how you
verified them. Commits follow Conventional Commits (`feat:`, `fix:`, `docs:`, ...).

## License

Released under the [MIT License](LICENSE). The challenge data belongs to its
owners and is not covered by this license.
