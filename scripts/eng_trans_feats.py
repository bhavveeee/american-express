import pandas as pd
from pathlib import Path

DATA_DIR = Path("D:/amexxx/data")

# Load base datasets
train = pd.read_parquet(DATA_DIR / "train_with_new_feats.parquet")
test = pd.read_parquet(DATA_DIR / "test_with_new_feats.parquet")
trans = pd.read_parquet(DATA_DIR / "add_trans.parquet")

# Convert IDs to string
for df in [train, test, trans]:
    for col in ["id2", "id3", "id5"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

# Ensure amount is numeric (f367)
trans["f367"] = pd.to_numeric(trans["f367"], errors="coerce")

# You don’t have quantity → just compute total & avg amount
agg_trans = trans.groupby("id2").agg(
    f370_total_amount=("f367", "sum"),
    f376_avg_amount=("f367", "mean")
).reset_index()

# Merge into train/test on id2
train = train.merge(agg_trans, on="id2", how="left")
test = test.merge(agg_trans, on="id2", how="left")

# Save updated datasets
train.to_parquet(DATA_DIR / "train_with_new_feats.parquet", index=False)
test.to_parquet(DATA_DIR / "test_with_new_feats.parquet", index=False)

print("✅ Transaction features (total and avg amount) added and saved.")
