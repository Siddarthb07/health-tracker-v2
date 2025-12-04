# migrate_db.py
import sqlite3

DB_NAME = "health_data.db"

REQUIRED_COLUMNS_USERS = {
    "sugar_drinks": "REAL DEFAULT 0",
    "fruit_servings": "REAL DEFAULT 1",
    "veg_servings": "REAL DEFAULT 2",
    "stress_level": "REAL DEFAULT 5",
    "systolic": "REAL DEFAULT 120",
    "diastolic": "REAL DEFAULT 80",
    "resting_hr": "REAL DEFAULT 70",
    "existing_diabetes": "INTEGER DEFAULT 0",
    "existing_cvd": "INTEGER DEFAULT 0",
    "existing_cancer": "INTEGER DEFAULT 0",
    "existing_asthma": "INTEGER DEFAULT 0",
    "family_history_cancer": "INTEGER DEFAULT 0",
    "family_history_cvd": "INTEGER DEFAULT 0",
    "family_history_diabetes": "INTEGER DEFAULT 0"
}

REQUIRED_COLUMNS_USER_DATA = {
    "sugar_drinks": "REAL DEFAULT 0",
    "fruit_servings": "REAL DEFAULT 0",
    "veg_servings": "REAL DEFAULT 0",
    "stress_level": "REAL DEFAULT 0"
}


def add_missing_columns(table, required_columns):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Get current columns in table
    c.execute(f"PRAGMA table_info({table})")
    existing_cols = [row[1] for row in c.fetchall()]

    # Add missing columns
    for col, col_def in required_columns.items():
        if col not in existing_cols:
            print(f"Adding column {col} to {table}")
            c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    add_missing_columns("users", REQUIRED_COLUMNS_USERS)
    add_missing_columns("user_data", REQUIRED_COLUMNS_USER_DATA)
    print("✅ Migration complete! All columns are now present.")
