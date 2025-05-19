import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    "user_id": range(10000),
    "playtime": np.random.exponential(10, 10000),
    "sessions": np.random.poisson(5, 10000),
    "purchases": np.random.poisson(1, 10000),
    "churned": np.random.binomial(1, 0.3, 10000)
})

df.to_csv("data/user_data.csv", index=False)
print("✅ Sample data generated at data/user_data.csv")
