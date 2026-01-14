# Maternal Health Risk Prediction

Privacy-first maternal risk prediction using federated learning. This service simulates multiple hospitals training a shared model without sharing raw patient data, then serves predictions and metrics via a Flask API.

## Features
- Federated learning coordinator + hospital nodes
- Differential privacy via Opacus
- Synthetic maternal health dataset generator
- REST API with training, evaluation, prediction, and metrics
- SQLite persistence for model history and prediction counts

## Quick Start
```bash
cd "/Users/hafsaghannaj/Desktop/Maternal Health Risk Prediction"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
```

Server runs on `http://localhost:5001`.

## API Endpoints
Base URL: `http://localhost:5001`

- `GET /` Health + overview page
- `GET /metrics` Metrics dashboard
- `GET /about` About page
- `POST /api/initialize` Initialize federated learning
- `POST /api/train` Train for N rounds (default in config)
- `GET /api/evaluate` Evaluate current model
- `POST /api/predict` Predict maternal risk
- `GET /api/history` Training history

### Example Requests
Initialize:
```bash
curl -X POST http://localhost:5001/api/initialize
```

Train (5 rounds):
```bash
curl -X POST http://localhost:5001/api/train -H "Content-Type: application/json" -d '{"rounds": 5}'
```

Predict:
```bash
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"patient_data": [25, 22, 120, 80, 90, 85, 12, 250, 6, 4, 20, 0.8, 2.0, 1.1, 7.2, 30, 9.5, 40, 4.5, 180, 140, 45, 95, 1, 2]}'
```

## Configuration
Edit `config.py` to adjust:
- number of hospitals and samples
- model architecture
- federated rounds and learning rate
- differential privacy settings

## Data
Synthetic data is generated with realistic feature ranges in:
- `app/data/synthetic_data.py`

## Storage
- SQLite DB: `artemis.sqlite3`
- Saved models: `saved_models/`

## Notes
This is a demo system using synthetic data. It is not a medical device and should not be used for clinical decisions.
