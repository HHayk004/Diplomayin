import numpy as np
import pandas as pd
import pyodbc

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def logistic_reg(alpha, X, y, iters=5000):
    theta = np.zeros(X.shape[1] + 1)

    for _ in range(iters):
        z = np.dot(X, theta[:-1]) + theta[-1]
        h = sigmoid(z)
        gradient = np.dot(X.T, h - y) / y.size

        theta[:-1] -= alpha * gradient
        theta[-1] -= alpha * (h - y).mean()

    return theta

conn = pyodbc.connect("Driver={ODBC Driver 18 for SQL Server};" \
                      "Server=tcp:mt240-sql-1.database.windows.net,1433;" \
                      "Database=mt240-sql-db-1;" \
                      "Uid=adminuser;" \
                      "Pwd=D13nam04295;" \
                      "Encrypt=yes;" \
                      "TrustServerCertificate=no;" \
                      "Connection Timeout=30;")

df = pd.read_sql("SELECT * FROM train_data", conn)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

theta = logistic_reg(0.01, X, y)

cursor = conn.cursor()
cursor.execute("TRUNCATE TABLE model_params")
conn.commit()

for i, w in enumerate(theta):
    cursor.execute(
        "INSERT INTO model_params (feature_index, weight) VALUES (?, ?)",
        i, float(w)
    )

conn.commit()
print("Model trained and saved")