# Vesta

Vesta is a menstrual cycle prediction project with a training pipeline and a small Flask web app for local use. It trains a Random Forest model and provides cycle start date predictions from recent history.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e packages/vesta_ml
```

### 2. Pipeline (preprocess → train → evaluate)

```bash
# Preprocess data (uses packages/vesta_ml/data/raw/cycles.csv or creates sample data)
python packages/vesta_ml/scripts/preprocess.py

# Train models (Random Forest + quantile models)
python packages/vesta_ml/scripts/train.py

# Evaluate and generate plots/metrics
python packages/vesta_ml/scripts/evaluate.py
```

### 3. Predict from CSV

```bash
python packages/vesta_ml/scripts/predict.py --input packages/vesta_ml/data/raw/cycles.csv
```

### 4. Run the web app (local)

```bash
python -m flask --app app:create_app --debug run
```

The app stores data in `instance/vesta.db` and predicts once six cycles are saved.

## Deployment

**Digital Ocean Droplet**
- [Complete Deployment Guide](./DEPLOYMENT.md) - $4-6/month
- Includes model training, Docker setup, and optional HTTPS with SSL
- Docker-based with persistent SQLite storage

## Data format

Your CSV should look like:

```csv
start_date,end_date
2023-01-15,2023-01-20
2023-02-12,2023-02-17
2023-03-10,2023-03-16
```

## Notes

- Trained model artifacts are saved under `packages/vesta_ml/models/random_forest/`.
- The scaler is saved under `packages/vesta_ml/data/processed/scaler.pkl`.
- This project is for educational purposes and is not medical advice.
