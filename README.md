# CDR — Credit Default Risk Model

CDR is a business-facing credit risk model: given a credit card applicant's
demographic and payment history data, it predicts their probability of
default and produces a risk-ranked, actionable output for a lending/risk
team — not a consumer-facing tool. Built in Python, SQL, scikit-learn, and
pandas.

## Why no front end

Unlike a consumer diagnostic tool, credit risk scoring isn't something an
individual applicant interacts with directly — it's a decision-support
system a risk/underwriting team runs against a portfolio of applicants in
bulk. The real deliverable is a model that outputs interpretable
probabilities and a ranked list a human analyst acts on, which is why this
project's "interface" is a batch scoring table and a threshold/policy
trade-off, not a web UI.

## The data

UCI Credit Card dataset: 30,000 clients, demographic features (credit
limit, sex, education, marriage, age), 6 months of payment status history
(`PAY_0`...`PAY_6`), 6 months of bill amounts, 6 months of payment amounts,
and a binary target (`default_flag`) — did the client default on their next
payment. ~22% of clients in the dataset defaulted.

## Architecture

1. **SQL data layer** — `01_sql_data_layer.py` loads the CSV into a SQLite
   database (`cdr.db`) and runs the core EDA as actual SQL queries: target
   distribution, default rate by education/marriage segment, summary stats
   by default status, and — the most predictive single cut — default rate
   by most recent payment delay (`PAY_0`). The model-ready feature table is
   then pulled via a SQL query, not a raw `pd.read_csv()`, so SQL is a real
   part of the pipeline, not a label.

2. **Model training & evaluation** — `02_model_training.py` trains a
   logistic regression classifier (scikit-learn) on the SQL-sourced data,
   with standardized features (`StandardScaler`) and an 80/20 stratified
   train/test split. Evaluates with ROC/AUC and reports standardized
   coefficients for interpretability.

3. **Threshold / lending policy trade-off table** — also in
   `02_model_training.py`: translates the model's raw probability output
   into a business decision table — at each probability threshold, what %
   of applicants get flagged, what % of true defaulters get caught
   (recall), and what fraction of flagged applicants actually default
   (precision). This is the literal "decision-support recommendation for
   lending policy" the project description calls for — a policy team would
   pick a threshold from this table based on their risk appetite, not from
   the AUC number alone.

4. **Batch risk scoring** — `03_batch_scoring.py` scores a batch of
   applicants at once (not one at a time) and outputs a risk-ranked table
   with a recommended action per applicant (Approve / Flag for Review /
   Decline), plus a portfolio-level summary count — the actual
   business-facing deliverable.

## Results

- **ROC AUC: 0.708** — a reasonable, honest result for this dataset (not
  inflated); credit default from payment history alone is a genuinely hard
  prediction problem.
- **Most predictive feature: `PAY_0`** (most recent payment delay status).
  Default rate jumps from ~13% for clients current on payments to ~69-78%
  for clients 2+ months delinquent — by far the strongest single signal in
  the data.
- **Threshold trade-off** (selected rows):

  | Threshold | % applicants flagged | % true defaulters caught | Precision |
  |---|---|---|---|
  | 0.2 | 47.1% | 67.9% | 31.9% |
  | 0.3 | 18.3% | 45.9% | 55.5% |
  | 0.5 | 7.7% | 24.0% | 68.7% |

  Lowering the threshold catches more real defaulters but flags far more
  good customers too — this is the actual trade-off a lending policy
  decision is made against.

## An honest limitation, visible in the batch scoring output

Running the batch scorer on a random sample of 25 applicants surfaces two
cases where the model recommended "Approve" but the applicant actually
defaulted — real false negatives, left visible rather than cherry-picked
away. This is expected at AUC 0.708 (a good but imperfect model) and is
useful to discuss directly: it's exactly why a real deployment would also
need ongoing monitoring and periodic retraining, not a one-time model.

## How to run

```bash
python 01_sql_data_layer.py    # loads CSV into SQLite, runs SQL EDA
python 02_model_training.py    # trains model, ROC/AUC, threshold policy table
python 03_batch_scoring.py     # batch-scores a sample of applicants
```

## Stack

Python (pandas, scikit-learn — LogisticRegression, StandardScaler,
train_test_split, roc_auc_score), SQL (SQLite — aggregations, GROUP BY,
CASE-free conditional aggregation via SUM/AVG on the binary target),
matplotlib (ROC curve).

## Possible extensions

- Try alternative models (random forest, gradient boosting) and compare AUC
- Address class imbalance explicitly (SMOTE, class weighting) rather than
  relying on the model implicitly handling a 78/22 split
- Move from a single train/test split to k-fold cross-validation for a more
  robust AUC estimate
