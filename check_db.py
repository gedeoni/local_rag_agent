import lancedb
import pandas as pd

LANCEDB_URI = "./.lancedb"
TABLE_NAME = "deepseek_rag_table"

try:
    db = lancedb.connect(LANCEDB_URI)
    table_names = db.table_names()
    print(f"Tables found: {table_names}")

    if TABLE_NAME in table_names:
        tbl = db.open_table(TABLE_NAME)
        print(f"Total rows in '{TABLE_NAME}': {len(tbl)}")
        
        if len(tbl) > 0:
            # Convert the first 5 rows to a pandas DataFrame for easy viewing
            df = tbl.to_pandas().head(5)
            print("\nFirst 5 rows:")
            print(df)
        else:
            print(f"The table '{TABLE_NAME}' is empty.")
    else:
        print(f"Table '{TABLE_NAME}' does not exist yet.")

except Exception as e:
    print(f"Error connecting to LanceDB: {e}")
