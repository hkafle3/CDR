"""
CDR — Credit Default Risk Model
Step 3: Batch Risk Scoring (Business-Facing Output)

Unlike a consumer-facing tool, CDR's real output is a scored, ranked list of
applicants for a risk/underwriting team to act on -- not a single result for
an individual user. This script scores an entire batch of applicants at once
and produces a risk-ranked table, which is the actual deliverable a lending
team would use, plus a simple recommended action per applicant based on the
threshold policy table from 02_model_training.py.
"""

import sqlite3
import pandas as pd
import importlib.util

_spec = importlib.util.spec_from_file_location("sql_layer", "01_sql_data_layer.py")
sql_layer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sql_layer)

_spec2 = importlib.util.spec_from_file_location("training", "02_model_training.py")
training = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(training)

DB_PATH = "cdr.db"

# Decision bands, informed by the threshold/policy table in 02_model_training.py
def recommend_action(prob):
    if prob >= 0.5:
        return "DECLINE / MANUAL REVIEW"
    elif prob >= 0.3:
        return "FLAG FOR REVIEW"
    else:
        return "APPROVE"


def score_batch(df, model, scaler, n=25, random_state=7):
    """Scores a random batch of n applicants and ranks them by risk."""
    sample = df.sample(n=n, random_state=random_state).copy()
    X_sample = sample[training.FEATURES]
    X_scaled = scaler.transform(X_sample)
    sample["predicted_default_probability"] = model.predict_proba(X_scaled)[:, 1].round(4)
    sample["recommended_action"] = sample["predicted_default_probability"].apply(recommend_action)
    sample["actual_outcome"] = sample["default_flag"].map({0: "did not default", 1: "defaulted"})

    ranked = sample.sort_values("predicted_default_probability", ascending=False)
    return ranked[["LIMIT_BAL", "AGE", "PAY_0", "predicted_default_probability",
                   "recommended_action", "actual_outcome"]]


def portfolio_summary(scored_df):
    """Rolls up the batch into a quick portfolio-level view, the kind of
    summary a risk manager would glance at first."""
    summary = scored_df["recommended_action"].value_counts().reset_index()
    summary.columns = ["recommended_action", "n_applicants"]
    return summary


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    df = sql_layer.get_model_ready_table(conn)
    conn.close()

    results = training.train_and_evaluate(df)

    print("\n=== Batch Risk Scoring: 25 Applicants ===")
    scored = score_batch(df, results["model"], results["scaler"], n=25)
    print(scored.to_string(index=False))

    print("\n=== Portfolio Summary ===")
    print(portfolio_summary(scored).to_string(index=False))
