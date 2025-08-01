import pandas as pd
from pathlib import Path

print("🚀 Starting to build train_data2...")

DATA_DIR = Path("D:/amexxx/data")

# Load base files
train = pd.read_parquet(DATA_DIR / "train_fixed.parquet")
test = pd.read_parquet(DATA_DIR / "test_data.parquet")
offer = pd.read_parquet(DATA_DIR / "offer_metadata.parquet")
event = pd.read_parquet(DATA_DIR / "add_event.parquet")
trans = pd.read_parquet(DATA_DIR / "add_trans.parquet")

print("✅ Loaded all parquet files.")

# Standardize key types
for df in [train, test, offer, event, trans]:
    for col in ["id2", "id3", "id5"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

# Merge offer metadata (id3 → id5)
offer = offer.rename(columns={"id3": "id3_offer"})
train = train.merge(offer, left_on="id3", right_on="id3_offer", how="left")
test = test.merge(offer, left_on="id3", right_on="id3_offer", how="left")
print("🔗 Merged offer_metadata.")

# Add event_count feature
event_counts = event.groupby(["id2", "id3"]).size().reset_index(name="event_count")
train = train.merge(event_counts, on=["id2", "id3"], how="left")
test = test.merge(event_counts, on=["id2", "id3"], how="left")
train["event_count"] = train["event_count"].fillna(0)
test["event_count"] = test["event_count"].fillna(0)
print("📊 Added event_count feature.")

# Prepare transaction features
trans = trans.rename(columns={"f371": "amount", "f368": "quantity"})
trans["amount"] = pd.to_numeric(trans["amount"], errors="coerce")
trans["quantity"] = pd.to_numeric(trans["quantity"], errors="coerce")

# Aggregate transaction features by user only (id2)
trans_agg = trans.groupby("id2").agg(
    trans_count=("amount", "count"),
    trans_sum_amount=("amount", "sum"),
    trans_avg_amount=("amount", "mean"),
    trans_max_amount=("amount", "max"),
    trans_sum_quantity=("quantity", "sum"),
).reset_index()

# Merge transaction features
train = train.merge(trans_agg, on="id2", how="left")
test = test.merge(trans_agg, on="id2", how="left")

# Fill NaNs for transaction stats
trans_cols = [
    "trans_count", "trans_sum_amount", "trans_avg_amount",
    "trans_max_amount", "trans_sum_quantity"
]
for col in trans_cols:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

print("💸 Added transaction-based features (amount + quantity).")

# Final check and save
train.to_parquet(DATA_DIR / "train_data2.parquet", index=False)
test.to_parquet(DATA_DIR / "test_data2.parquet", index=False)

print(f"✅ Saved train_data2.parquet: {train.shape}")
print(f"✅ Saved test_data2.parquet:  {test.shape}")
