import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .config import SIMILARITY_THRESHOLD, MAX_RESULTS
from .llm import call_monster_api

def recommend_tests(user_query, model, df, embeddings):
    query_embedding = model([user_query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    sorted_indices = similarities.argsort()[::-1]
    filtered_indices = [i for i in sorted_indices if similarities[i] >= SIMILARITY_THRESHOLD]
    if not filtered_indices:
        filtered_indices = [sorted_indices[0]]
    top_indices = filtered_indices[:MAX_RESULTS]
    results = df.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]
    return results

def generate_llm_response(user_query, recommendations_df, client):
    context = "You are an assessment recommendation assistant for SHL. Summarize the most relevant tests for the user's need."
    top_items = []
    for _, row in recommendations_df.iterrows():
        time = f", {row['Assessment Length']}" if pd.notnull(row['Assessment Length']) else ""
        top_items.append(f"- {row['name']} ({row['test_type']}{time})")
    message = f"""
User Query: \"{user_query}\"

Based on SHL's test catalog, here are the most relevant assessments:

{chr(10).join(top_items)}

Please summarize this in 3-4 sentences.
"""
    return call_monster_api(message, context, client)
