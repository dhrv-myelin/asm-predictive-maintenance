import psycopg2
import csv

# ⚠️ Update these connection details
DB_CONFIG = {
    "host": "localhost",       # your DB host
    "port": 5432,
    "dbname": "glue-dispenser-db",
    "user": "postgres",        # your DB user
    "password": "postgres" # your DB password
}

TABLE_NAME = "baseline_metrics"  # ⚠️ Replace with your actual table name
OUTPUT_FILE = "export.csv"

def export_to_csv():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute(f"SELECT * FROM {TABLE_NAME} ORDER BY id;")
    rows = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)  # header
        writer.writerows(rows)

    print(f"✅ Exported {len(rows)} rows to {OUTPUT_FILE}")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    export_to_csv()