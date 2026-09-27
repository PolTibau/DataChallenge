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

# ── Transformer v4 ────────────────────────────────────────────
# Prova ràpida (2 èpoques, poques corbes)
LSTM_DEBUG=1 python -m lstm.main_train_combined

# Entrenar (es desa a lstm/checkpoints/v4/seed42/)
python -m lstm.main_train_combined
# Entrenaments addicionals per a l'ensemble (opcional)
python -m lstm.main_train_combined --seed 43
# Variant amb un token per mitja hora (es desa a seed44_p1/, ~4x més lent)
LSTM_PATCH_SIZE=1 LSTM_BATCH_SIZE=32 python -m lstm.main_train_combined --seed 44
# Reprendre un entrenament interromput
python -m lstm.main_train_combined --seed 42 --resume

# Submissió: combina tots els seed*/ → outputs/lstm/y_pred_test_v4.csv
python -m lstm.main_predict
```

## Models

### KNN
Cerca les K corbes de referència més similars (usant distància Manhattan
sobre els timesteps vàlids) i omple els forats amb una mitjana ponderada
per rang. Ràpid i interpretable, però tracta cada timestep de forma independent.

### Transformer v4 (Temporal Masked Autoencoder)
Entrena sobre les ~58.000 corbes de referència (train + test) amb forats
simulats que imiten els reals. Per a cada mitja hora rep, a més del valor,
16 canals de context (`lstm/src/context.py`): màscara, interpolació, mateixa
hora ±1 dia/setmana, perfil habitual del consumidor, corba de població i
corba dels 50 consumidors més semblants. La pèrdua es pondera per l'escala
de cada corba (≈ MAE en watts) i el millor model es tria amb les 999 corbes
holed_ de train etiquetades. La predicció promitja els millors checkpoints
de tots els entrenaments i barreja amb interpolació als forats d'1-2 punts.

## Mètrica
MAE en watts, calculat NOMÉS sobre els punts que eren NaN originalment.
