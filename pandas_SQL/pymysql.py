import pandas as pd
import pymysql

conn = pymysql.connect(
    host="localhost",
    user= "root",
    password="root",
    database="world"
)

quary = "select * from country"

df = pd.read_sql(quary,conn)

print(df)

conn.close()