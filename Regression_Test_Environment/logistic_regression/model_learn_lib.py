import numpy as np
import pandas as pd
import pyodbc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

df = pd.read_sql("SELECT * FROM train_data", conn)

if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.iloc[:, :-1]
y = df.iloc[:, -1].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=5000, solver='lbfgs')
model.fit(X_scaled, y)

weights = model.coef_[0]
bias = model.intercept_[0]

cursor = conn.cursor()
cursor.execute("TRUNCATE TABLE model_params")
conn.commit()

for i, w in enumerate(weights):
    cursor.execute(
        "INSERT INTO model_params (feature_index, weight, is_bias) VALUES (?, ?, 0)",
        int(i),
        float(w)
    )

cursor.execute(
    "INSERT INTO model_params (feature_index, weight, is_bias) VALUES (?, ?, 1)",
    -1,
    float(bias)
)

conn.commit()

print("Model trained and saved to SQL successfully")