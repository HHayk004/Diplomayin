import numpy as np
import pandas as pd
import pyodbc
from sklearn.metrics import accuracy_score

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

conn = pyodbc.connect("Driver={ODBC Driver 18 for SQL Server};" \
                      "Server=tcp:mt240-sql-1.database.windows.net,1433;" \
                      "Database=mt240-sql-db-1;" \
                      "Uid=adminuser;" \
                      "Pwd=D13nam04295;" \
                      "Encrypt=yes;" \
                      "TrustServerCertificate=no;" \
                      "Connection Timeout=30;")

df = pd.read_sql("SELECT * FROM test_data", conn)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

model_df = pd.read_sql("SELECT * FROM model_params ORDER BY feature_index", conn)
theta = model_df["weight"].values

z = np.dot(X, theta[:-1]) + theta[-1]
probs = sigmoid(z)
preds = (probs >= 0.5).astype(int)

acc = accuracy_score(y, preds)
print("Accuracy:", acc)

cursor = conn.cursor()
cursor.execute("TRUNCATE TABLE pvalues")
conn.commit()

for i in range(len(probs)):
    cursor.execute(
        "INSERT INTO pvalues (actual, predicted, probability) VALUES (?, ?, ?)",
        int(y[i]), int(preds[i]), float(probs[i])
    )

conn.commit()
print("P-values saved")