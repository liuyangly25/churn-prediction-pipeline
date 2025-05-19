import dask.dataframe as dd
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import ray
from ray.train.xgboost import XGBoostTrainer
from ray.train import ScalingConfig

# Load data
df = dd.read_csv("data/user_data.csv")
df['log_playtime'] = df['playtime'].map_partitions(lambda x: np.log1p(x))

X = df[['log_playtime', 'sessions', 'purchases']].compute()
y = df['churned'].compute()

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = X_train.copy()
train_df["churned"] = y_train
val_df = X_val.copy()
val_df["churned"] = y_val

# Initialize Ray
ray.init()

trainer = XGBoostTrainer(
    label_column="churned",
    params={"objective": "binary:logistic", "eval_metric": "logloss"},
    scaling_config=ScalingConfig(num_workers=2),
    datasets={
        "train": ray.data.from_pandas(train_df),
        "valid": ray.data.from_pandas(val_df)
    },
)

result = trainer.fit()
print("Training complete. Validation logloss:", result.metrics['valid-logloss'])
