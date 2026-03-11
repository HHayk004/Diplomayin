import pandas as pd
import sys

try:
    df = pd.read_csv('defaultofcreditcardclients.csv', header=None)
except Exception as e:
    print(f"Error loading CSV: {e}")
    sys.exit(1)

# drop first 2 rows
df = df.iloc[2:].reset_index(drop=True)

# drop first column
df = df.iloc[:, 1:]

# save without headers
df.to_csv('defaultofcreditcardclients_cleaned.csv', index=False, header=False)

print("Cleaned file saved without headers and ID column")
