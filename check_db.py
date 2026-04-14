import lancedb
import pandas as pd
import argparse
import sys

LANCEDB_URI = "./.lancedb"
DOCUMENTS_VECTOR_TABLE = "aura_farm_vectors"
DOCUMENTS_REGISTRY_TABLE = "document_registry"

def main():
    parser = argparse.ArgumentParser(description="Check or clear the LanceDB database.")
    parser.add_argument("--clear", action="store_true", help="Clear (drop) the RAG table.")
    args = parser.parse_args()

    try:
        db = lancedb.connect(LANCEDB_URI)
        table_names = db.table_names()

        if args.clear:
            cleared = False
            for t_name in [DOCUMENTS_VECTOR_TABLE, "openai_rag_table", DOCUMENTS_REGISTRY_TABLE]:
                if t_name in table_names:
                    db.drop_table(t_name)
                    print(f"✅ Table '{t_name}' has been cleared.")
                    cleared = True
            
            if not cleared:
                print("ℹ️ No relevant RAG tables or registries found to clear.")
            return

        print(f"Tables found: {table_names}")

        if DOCUMENTS_VECTOR_TABLE in table_names:
            tbl = db.open_table(DOCUMENTS_VECTOR_TABLE)
            print(f"Total rows in '{DOCUMENTS_VECTOR_TABLE}': {len(tbl)}")
            
            if len(tbl) > 0:
                # Convert the first 5 rows to a pandas DataFrame for easy viewing
                df = tbl.to_pandas().head(5)
                print("\nFirst 5 rows:")
                print(df)
            else:
                print(f"The table '{DOCUMENTS_VECTOR_TABLE}' is empty.")
        else:
            print(f"Table '{DOCUMENTS_VECTOR_TABLE}' does not exist yet.")

    except Exception as e:
        print(f"Error connecting to LanceDB: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
