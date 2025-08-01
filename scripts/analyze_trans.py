import pandas as pd
from pathlib import Path

# Path to data directory
DATA_DIR = Path("D:/amexxx/data")

# Load the add_trans.parquet file
trans = pd.read_parquet(DATA_DIR / "add_trans.parquet")

# Basic info
print("\n🔍 Shape:", trans.shape)
print("\n📋 Columns:", trans.columns.tolist())
print("\n📊 Data Types:")
print(trans.dtypes)

# Summary stats
print("\n📈 Summary statistics (numerical):")
print(trans.describe())

# Check nulls
print("\n❓ Missing values:")
print(trans.isnull().sum())

# Check unique values per column
print("\n🔢 Unique values:")
for col in trans.columns:
    print(f"{col}: {trans[col].nunique()}")

# Show a sample of rows
print("\n👀 Sample rows:")
print(trans.head(10))

# Check value counts for key categorical features (if any)
cat_cols = trans.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    print(f"\n📦 Value counts for {col}:")
    print(trans[col].value_counts().head(10))
