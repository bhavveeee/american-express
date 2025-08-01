import pandas as pd

# Define input/output paths
data_path = "D:/amexxx/data"  # Use forward slashes or raw strings for Windows paths

train_parquet = f"{data_path}/train_with_new_feats_v2.parquet"
test_parquet = f"{data_path}/test_with_new_feats_v2.parquet"

train_csv = f"{data_path}/train_data_v2.csv"
test_csv = f"{data_path}/test_data_v2.csv"

# Convert train
print("🔄 Loading train_data_v2.parquet...")
train_df = pd.read_parquet(train_parquet)
print("💾 Saving train_data_v2.csv...")
train_df.to_csv(train_csv, index=False)

# Convert test
print("🔄 Loading test_data_v2.parquet...")
test_df = pd.read_parquet(test_parquet)
print("💾 Saving test_data_v2.csv...")
test_df.to_csv(test_csv, index=False)

print("✅ All conversions completed.")
