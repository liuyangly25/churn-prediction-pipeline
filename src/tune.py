import ray
from ray import tune
from ray.train.xgboost import XGBoostTrainer
from ray.train import ScalingConfig

from sklearn.model_selection import train_test_split
import pandas as pd
import dask.dataframe as dd
import numpy as np

def get_dataset():
    df = dd.read_csv("data/user_data.csv")
    df['log_playtime'] = df['playtime'].map_partitions(lambda x: np.log1p(x))

    X = df[['log_playtime', 'sessions', 'purchases']].compute()
    y = df['churned'].compute()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_df = X_train.copy()
    train_df["churned"] = y_train
    val_df = X_val.copy()
    val_df["churned"] = y_val

    return ray.data.from_pandas(train_df), ray.data.from_pandas(val_df)

def train_xgb(config):
    train_ds, val_ds = get_dataset()

    trainer = XGBoostTrainer(
        label_column="churned",
        params={
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": config["eta"],
            "max_depth": config["max_depth"],
            "subsample": config["subsample"],
        },
        scaling_config=ScalingConfig(num_workers=2),
        datasets={"train": train_ds, "valid": val_ds},
    )

    result = trainer.fit()
    tune.report(logloss=result.metrics["valid-logloss"])

if __name__ == "__main__":
    ray.init()

    search_space = {
        "eta": tune.loguniform(1e-3, 0.3),
        "max_depth": tune.randint(3, 10),
        "subsample": tune.uniform(0.5, 1.0),
    }

    tuner = tune.Tuner(
        train_xgb,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="logloss",
            mode="min",
            num_samples=10,
        ),
    )

    results = tuner.fit()
    print("🎯 Best config found:")
    print(results.get_best_result().config)
