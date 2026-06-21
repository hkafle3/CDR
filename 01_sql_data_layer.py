"""
CDR — Credit Default Risk Model
Step 1: SQL Data Layer

Loads the UCI Credit Card dataset into a SQLite database and performs the
core EDA / feature engineering aggregations as actual SQL queries, rather
than purely in pandas. This is the layer that makes "SQL" in the project's
tech stack a real, substantive part of the pipeline, not just a label.

The dataset: 30,000 credit card clients, demographic info, 6 months of
payment status / bill amount / payment amount history, and a binary target
(did this client default on their next payment).
"""

import sqlite3
import pandas as pd

CSV_PATH = "UCI_Credit_Card.csv"
DB_PATH = "cdr.db"


def load_csv_into_sql(conn):
    df = pd.read_csv(CSV_PATH)
    # Real column name has dots in it; SQL-friendly rename for clarity.
    df = df.rename(columns={"default.payment.next.month": "default_flag"})
    df.to_sql("credit_clients", conn, if_exists="replace", index=False)
    print(f"Loaded {len(df)} rows into 'credit_clients' table.")
    return df


def eda_queries(conn):
    """
    Core EDA, done in SQL rather than pandas .value_counts() / .describe():
    target distribution, default rate by demographic segment, and summary
    stats on credit limit and age by default status.
    """
    print("\n--- Target distribution ---")
    print(pd.read_sql_query("""
        SELECT default_flag,
               COUNT(*) AS n_clients,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM credit_clients), 1) AS pct
        FROM credit_clients
        GROUP BY default_flag
    """, conn).to_string(index=False))

    print("\n--- Default rate by education level ---")
    print(pd.read_sql_query("""
        SELECT EDUCATION,
               COUNT(*) AS n_clients,
               ROUND(100.0 * SUM(default_flag) / COUNT(*), 1) AS default_rate_pct
        FROM credit_clients
        GROUP BY EDUCATION
        ORDER BY EDUCATION
    """, conn).to_string(index=False))

    print("\n--- Default rate by marriage status ---")
    print(pd.read_sql_query("""
        SELECT MARRIAGE,
               COUNT(*) AS n_clients,
               ROUND(100.0 * SUM(default_flag) / COUNT(*), 1) AS default_rate_pct
        FROM credit_clients
        GROUP BY MARRIAGE
        ORDER BY MARRIAGE
    """, conn).to_string(index=False))

    print("\n--- Credit limit & age summary stats by default status ---")
    print(pd.read_sql_query("""
        SELECT default_flag,
               COUNT(*) AS n_clients,
               ROUND(AVG(LIMIT_BAL), 0) AS avg_credit_limit,
               ROUND(AVG(AGE), 1) AS avg_age,
               ROUND(AVG(BILL_AMT1), 0) AS avg_latest_bill,
               ROUND(AVG(PAY_AMT1), 0) AS avg_latest_payment
        FROM credit_clients
        GROUP BY default_flag
    """, conn).to_string(index=False))

    print("\n--- Default rate by recent payment delay (PAY_0) ---")
    print(pd.read_sql_query("""
        SELECT PAY_0,
               COUNT(*) AS n_clients,
               ROUND(100.0 * SUM(default_flag) / COUNT(*), 1) AS default_rate_pct
        FROM credit_clients
        GROUP BY PAY_0
        ORDER BY PAY_0
    """, conn).to_string(index=False))


def get_model_ready_table(conn):
    """
    Pulls the full feature set + target via a single SQL query, which is what
    feeds the train/test split downstream -- the model is trained on data
    that came out of SQL, not a raw CSV read.
    """
    query = """
        SELECT
            LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
            PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
            BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
            PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6,
            default_flag
        FROM credit_clients
    """
    return pd.read_sql_query(query, conn)


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    load_csv_into_sql(conn)
    eda_queries(conn)
    model_df = get_model_ready_table(conn)
    print(f"\nModel-ready table pulled via SQL: {model_df.shape}")
    conn.close()
