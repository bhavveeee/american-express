import pandas as pd
from pathlib import Path

print("🧾 Loading data...")
DATA_DIR = Path("D:/amexxx/data")

# Load datasets
train = pd.read_parquet(DATA_DIR / "train_with_new_feats_v2.parquet")
test = pd.read_parquet(DATA_DIR / "test_with_new_feats_v2.parquet")
trans = pd.read_parquet(DATA_DIR / "add_trans.parquet")
offer = pd.read_parquet(DATA_DIR / "offer_metadata.parquet")

# Standardize column names and types
offer = offer.rename(columns={"id3": "id5"})
for df in [train, test, offer, trans]:
    for col in ["id2", "id3", "id5"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

# Parse timestamps
offer["offer_start"] = pd.to_datetime(offer["id12"], errors="coerce")
trans["trans_time"] = pd.to_datetime(trans["id8"], errors="coerce")

# Ensure trans has id3 by merging from train
id2_id3_map = train[["id2", "id3"]].drop_duplicates()
trans = trans.merge(id2_id3_map, on="id2", how="left")

# Get offer reference mapping for each (id2, id3)
offer_ref = train[["id2", "id3", "id5"]].drop_duplicates().merge(
    offer[["id5", "offer_start"]], on="id5", how="left"
)

# Filter pre-offer transactions
trans = trans.merge(offer_ref, on=["id2", "id3"], how="left")
trans = trans[trans["trans_time"] < trans["offer_start"]]

# Group by (id2, id3) to compute f374 and f378
agg_trans = trans.groupby(["id2", "id3"]).agg(
    f374=("f374", "count"),
    f378=("f370", "sum")
).reset_index()

# Merge into train/test
train = train.merge(agg_trans, on=["id2", "id3"], how="left")
test = test.merge(agg_trans, on=["id2", "id3"], how="left")

# Fill missing with 0
train[["f374", "f378"]] = train[["f374", "f378"]].fillna(0)
test[["f374", "f378"]] = test[["f374", "f378"]].fillna(0)

# Save back
train.to_parquet(DATA_DIR / "train_with_new_feats_v2.parquet", index=False)
test.to_parquet(DATA_DIR / "test_with_new_feats_v2.parquet", index=False)

print("✅ f374 and f378 added and saved into train_with_new_feats_v2.parquet and test_with_new_feats_v2.parquet.")
