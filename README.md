# Enedis — Imputació de Corbes de Consum Elèctric

Dos models comparats per al challenge d'imputació de corbes sintètiques d'Enedis.

## Estructura

```
data_challenge/
│
├── data/                        ← CSV compartits entre els dos models
│   ├── X_train.csv
│   ├── y_train.csv
│   └── X_test.csv
│
├── outputs/
│   ├── knn/                     ← prediccions KNN
│   └── lstm/                    ← prediccions Transformer
│
├── src/                         ← mòduls compartits
│   ├── data_loader.py           ← càrrega CSV + save_submission
│   └── evaluator.py             ← MAE, visualitzacions, comparació models
│
├── knn/                         ← model KNN
│   ├── src/
│   │   ├── config.py            ← paràmetres KNN
│   │   ├── preprocessor.py      ← normalització
│   │   └── knn_imputer.py       ← algorisme KNN
│   ├── main_train.py
│   ├── main_predict.py
│   └── optimize_hyperparams.py  ← Bayesian optimization (Optuna)
│
├── lstm/                        ← model Transformer
│   ├── src/
│   │   ├── config.py            ← paràmetres Transformer
│   │   ├── dataset.py           ← PyTorch Dataset amb masking
│   │   ├── model.py             ← arquitectura Transformer
│   │   └── trainer.py           ← entrenament + early stopping
│   ├── checkpoints/             ← pesos del model
│   ├── main_train.py
│   └── main_predict.py
│
├── requirements.txt
└── README.md
```

## Instal·lació

```bash
# PyTorch amb CUDA (GPU NVIDIA)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Resta de dependències
pip install -r requirements.txt
```

## Ús

Tots els scripts s'executen des de l'arrel `data_challenge/`:

```bash
# ── KNN ──────────────────────────────────────────────────────
# Optimitzar hiperparàmetres (opcional, ~1h)
python -m knn.optimize_hyperparams

# Avaluar sobre train + guardar prediccions
python -m knn.main_train

# Generar submissió test
python -m knn.main_predict

# ── Transformer ───────────────────────────────────────────────
# Entrenar (~30-60 min amb GPU)
python -m lstm.main_train

# Generar submissió test
python -m lstm.main_predict
```

## Models

### KNN
Cerca les K corbes de referència més similars (usant distància Manhattan
sobre els timesteps vàlids) i omple els forats amb una mitjana ponderada
per rang. Ràpid i interpretable, però tracta cada timestep de forma independent.

### Transformer (Temporal Masked Autoencoder)
Aprèn la dinàmica temporal de les corbes entrenant sobre les 20.000 corbes
de referència senceres. Usa masking aleatori durant l'entrenament per aprendre
a reconstruir forats. L'atenció global permet capturar dependències de llarg
termini (patrons diaris, setmanals).

## Mètrica
MAE en watts, calculat NOMÉS sobre els punts que eren NaN originalment.
