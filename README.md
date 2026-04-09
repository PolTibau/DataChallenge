# Enedis — KNN Imputation of Electric Load Curves

Solució al challenge d'imputació de corbes de consum elèctric sintètiques d'Enedis.

## Estructura

```
enedis-imputation/
├── data/                        # Dades (no pujar al repo)
│   ├── X_train.parquet
│   ├── y_train.parquet
│   ├── X_test.parquet
│   └── new_output_sample.parquet
├── src/
│   ├── config.py                # Paràmetres globals
│   ├── data_loader.py           # Càrrega de parquets
│   ├── preprocessor.py          # Normalització i anàlisi de forats
│   ├── knn_imputer.py           # Algorisme KNN principal
│   └── evaluator.py             # MAE i visualitzacions
├── outputs/                     # Prediccions generades
├── notebooks/
│   └── exploration.ipynb
├── main_train.py                # Avaluació sobre train (amb MAE)
├── main_predict.py              # Predicció final sobre test
└── requirements.txt
```

## Instal·lació

```bash
pip install -r requirements.txt
```

## Ús

### 1. Avaluació sobre train
```bash
python main_train.py
```
Genera `outputs/y_pred_train.parquet` i imprimeix el MAE vs. ground truth.

### 2. Predicció final (submissió)
```bash
python main_predict.py
```
Genera `outputs/y_pred_test.parquet` llest per enviar.

## Algorisme

Per a cada corba `holed_i` amb forats:
1. Calcula la distància euclidiana vs. totes les corbes de referència, usant **només** els timesteps vàlids (no-NaN).
2. Selecciona els **K veïns** més propers.
3. Omple cada forat amb la **mitjana ponderada** (pes = 1/distància) dels K veïns.

## Paràmetres ajustables (`src/config.py`)

| Paràmetre        | Default     | Descripció                              |
|------------------|-------------|-----------------------------------------|
| `K_NEIGHBORS`    | 10          | Nombre de veïns                         |
| `MIN_OVERLAP`    | 50          | Timesteps mínims comuns per distància   |
| `DISTANCE_METRIC`| euclidean   | `euclidean` o `cosine`                  |
| `WEIGHTING`      | inverse     | `inverse` (1/dist) o `uniform`          |
| `NORMALIZE`      | True        | Normalitzar abans de calcular distàncies|
| `NORM_METHOD`    | zscore      | `zscore` o `minmax`                     |
