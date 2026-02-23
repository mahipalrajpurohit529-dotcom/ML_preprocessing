import mysql.connector
import pandas as pd

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user= "root",
    password="root",
    database="world"
)


# SQL query
query = "SELECT * FROM country"

# Load directly into pandas
df = pd.read_sql(query, conn)

print(df)

conn.close()