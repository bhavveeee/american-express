import pandas as pd
trans = pd.read_parquet("D:/amexxx/data/add_trans.parquet")
print("📦 Columns in add_trans.parquet:", trans.columns.tolist())
