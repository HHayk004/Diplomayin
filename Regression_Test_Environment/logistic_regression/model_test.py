import numpy as np
import pandas as pd
import pyodbc
from sklearn.metrics import accuracy_score

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


conn = pyodbc.connect(
    "Driver={ODBC Driver 18 for SQL Server};"
    "Server=tcp:mt240-sql-1.database.windows.net,1433;"
    "Database=mt240-sql-db-1;"
    "Uid=adminuser;"
    "Pwd=D13nam04295;"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Connection Timeout=30;"
)

# load test data (already split in SQL)
df = pd.read_sql("SELECT * FROM test_data", conn)

# remove ID if exists
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# load model correctly
model_df = pd.read_sql("""
SELECT feature_index, weight, is_bias
FROM model_params
""", conn)

weights = model_df[model_df["is_bias"] == 0].sort_values("feature_index")["weight"].values
bias = model_df[model_df["is_bias"] == 1]["weight"].values[0]

# prediction
z = np.dot(X, weights) + bias
probs = sigmoid(z)
preds = (probs >= 0.5).astype(int)

acc = accuracy_score(y, preds)
print("Accuracy:", acc)

# save predictions
cursor = conn.cursor()
cursor.execute("TRUNCATE TABLE pvalues")
conn.commit()

for i in range(len(y)):
    cursor.execute(
        "INSERT INTO pvalues (actual, predicted, probability) VALUES (?, ?, ?)",
        int(y[i]), int(preds[i]), float(probs[i])
    )

conn.commit()
print("P-values saved")