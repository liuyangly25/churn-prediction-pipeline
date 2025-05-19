# Churn Prediction ML Pipeline (Ray + Dask + XGBoost)

A minimal, Dockerized machine learning pipeline for churn prediction using:
- XGBoost for training
- Ray for distributed training
- Dask for preprocessing

## 🚀 Quick Start

```bash
# Generate sample data
docker run --rm -v $PWD:/app -w /app python:3.10 python data/generate_data.py

# Train the model
docker build -t churn-trainer .
docker run --rm churn-trainer
