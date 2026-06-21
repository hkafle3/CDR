"""
CDR — Credit Default Risk Model
Step 2: Model Training & Evaluation

Trains a logistic regression classifier on the SQL-sourced feature table
(from 01_sql_data_layer.py) to predict probability of credit card default.
Evaluates with ROC/AUC and reports interpretable coefficients.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

import importlib.util
_spec = importlib.util.spec_from_file_location("sql_layer", "01_sql_data_layer.py")
sql_layer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sql_layer)

DB_PATH = "cdr.db"

FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
TARGET = "default_flag"


def train_and_evaluate(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train_scaled, y_train)

    y_pred_prob = logreg.predict_proba(X_test_scaled)[:, 1]
    y_pred = logreg.predict(X_test_scaled)

    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC AUC: {round(auc, 3)}")
    print("\nClassification report (default threshold = 0.5):")
    print(classification_report(y_test, y_pred))

    coeff_df = pd.DataFrame({"Feature": FEATURES, "Beta": logreg.coef_[0]})
    coeff_df = coeff_df.sort_values(by="Beta", key=abs, ascending=False)
    print("\nTop 10 features by |coefficient| (standardized scale):")
    print(coeff_df.head(10).to_string(index=False))

    return {
        "model": logreg, "scaler": scaler,
        "X_test": X_test, "y_test": y_test,
        "y_pred_prob": y_pred_prob, "y_pred": y_pred,
        "auc": auc, "coeff_df": coeff_df,
    }


def plot_roc_curve(y_test, y_pred_prob, auc, save_path="roc_curve.png"):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})", color="#9c2c30", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Credit Default Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nSaved ROC curve to {save_path}")


def threshold_policy_table(y_test, y_pred_prob):
    """
    Translates the model's probability output into a business decision-support
    table: at different probability thresholds, how many applicants would be
    flagged/declined, and what fraction of true defaulters would be caught?
    This is the "decision-support recommendations for lending policy" piece.
    """
    rows = []
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        flagged = (y_pred_prob >= threshold).astype(int)
        n_flagged = flagged.sum()
        pct_flagged = 100 * n_flagged / len(flagged)
        true_defaults_caught = ((flagged == 1) & (y_test == 1)).sum()
        total_true_defaults = (y_test == 1).sum()
        recall = 100 * true_defaults_caught / total_true_defaults
        precision = 100 * true_defaults_caught / n_flagged if n_flagged > 0 else 0
        rows.append({
            "threshold": threshold,
            "pct_applicants_flagged": round(pct_flagged, 1),
            "pct_true_defaulters_caught": round(recall, 1),
            "precision_of_flagged": round(precision, 1),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    df = sql_layer.get_model_ready_table(conn)
    conn.close()

    results = train_and_evaluate(df)
    plot_roc_curve(results["y_test"], results["y_pred_prob"], results["auc"])

    print("\n--- Threshold / Lending Policy Trade-off Table ---")
    policy_table = threshold_policy_table(results["y_test"], results["y_pred_prob"])
    print(policy_table.to_string(index=False))
    print("\nInterpretation: lowering the threshold flags more applicants and catches")
    print("more true defaulters, but at the cost of flagging more good customers too")
    print("(lower precision) -- this table is what a lending policy decision would")
    print("actually be based on, not the AUC number alone.")
