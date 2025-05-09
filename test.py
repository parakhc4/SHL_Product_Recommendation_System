# 📘 SHL Evaluation Notebook Template (Single Test Debug)

import pandas as pd
import numpy as np
from recommender.embeddings import load_data, load_model
from recommender.logic import recommend_tests
from sklearn.metrics import average_precision_score

# ---- Setup ----
model = load_model()
df, embeddings = load_data()

# ---- Load Single Test Case from CSV ----
test_df = pd.read_csv("dataset/Test_DATA.csv")

# Choose first test case only
single_query = test_df.iloc[0]["Query"]
relevant_names = test_df[test_df["Query"] == single_query]["Assessments"].dropna().tolist()

print(f"\nQuery: {single_query}")
print(f"\nGround Truth Assessments:")
for a in relevant_names:
    print(f"- {a}")

# ---- Run Recommendation ----
recs = recommend_tests(single_query, model, df, embeddings)
top_recs = recs.head(5)

print("\nTop 5 Recommendations:")
for _, row in top_recs.iterrows():
    print(f"- {row['name']} (sim: {row['similarity']:.4f})")

# ---- Compute AP@K and Recall@K ----
top_names = top_recs["name"].str.lower().tolist()
def normalize(text):
    return text.lower().replace(" | shl", "").strip()

y_true = [
    1 if any(normalize(rel) in normalize(name) for rel in relevant_names) else 0
    for name in top_names
]
y_scores = top_recs["similarity"].tolist()

ap = average_precision_score(y_true, y_scores) if any(y_true) else 0.0
recall = sum(y_true) / len(relevant_names) if relevant_names else 0.0

print(f"\nAP@5: {ap:.4f}")
print(f"Recall@5: {recall:.4f}")
