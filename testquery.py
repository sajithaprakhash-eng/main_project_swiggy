import sqlite3
import pandas as pd

# 1. Connect to your SQLite database
# 1. Connect to instamart.db located in the root folder (one level up)
conn = sqlite3.connect("../instamart.db")

# 2. Open queries.sql
with open("queries.sql", "r", encoding="utf-8") as f:
    sql_script = f.read()
conn = sqlite3.connect("instamart.db")

# Split the file by semicolons to execute/test individual queries
queries = [q.strip() for q in sql_script.split(";") if q.strip()]

for i, query in enumerate(queries, 1):
    df_result = pd.read_sql_query(query, conn)
    print(f"--- Q{i} Output Sample ---")
    print(df_result.head(3))
    print(f"Total rows returned for Q{i}: {len(df_result)}\n")

conn.close()