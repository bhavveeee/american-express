import pandas as pd

# Load raw training data
print("📦 Loading raw_train.parquet...")
try:
    df = pd.read_parquet("D:/amexxx/data/train_data.parquet")
except Exception as e:
    print("❌ Failed to load raw_train.parquet:", e)
    exit()

# Check structure
print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("📋 Columns:", df.columns.tolist())

# Check target values
if "y" not in df.columns:
    print("❌ 'y' column is missing!")
else:
    print("🔢 Target 'y' value counts:")
    print(df["y"].value_counts(dropna=False))

    # Show examples
    print("\n✅ Sample clicked offers (y=1):")
    print(df[df["y"] == 1].head())

    print("\n🚫 Sample non-clicked offers (y=0):")
    print(df[df["y"] == 0].head())

# Check for nulls
print("\n🧼 Nulls in important columns:")
print(df[["id1", "id2", "id3", "id4", "id5", "y"]].isnull().sum())
