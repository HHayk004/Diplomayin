import pandas as pd
import pyodbc

conn = pyodbc.connect("Driver={ODBC Driver 18 for SQL Server};" \
                      "Server=tcp:mt240-sql-1.database.windows.net,1433;" \
                      "Database=mt240-sql-db-1;" \
                      "Uid=adminuser;" \
                      "Pwd=D13nam04295;" \
                      "Encrypt=yes;" \
                      "TrustServerCertificate=no;" \
                      "Connection Timeout=30;")

# Load full dataset
df = pd.read_sql("SELECT * FROM defaultofcreditcardclients ORDER BY ID", conn)

# Split
train_df = df.iloc[:25000]
test_df = df.iloc[25000:30000]

# Save to DB
cursor = conn.cursor()

cursor.execute("TRUNCATE TABLE train_data")
cursor.execute("TRUNCATE TABLE test_data")
conn.commit()

# Insert (simple version)
for _, row in train_df.iterrows():
    cursor.execute("INSERT INTO train_data VALUES ({})".format(
        ",".join(["?"] * len(row))
    ), tuple(row))

for _, row in test_df.iterrows():
    cursor.execute("INSERT INTO test_data VALUES ({})".format(
        ",".join(["?"] * len(row))
    ), tuple(row))

conn.commit()
print("Train/Test split completed")