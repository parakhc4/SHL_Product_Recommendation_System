import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
# from settings import MONSTER_API_KEY
MONSTER_API_KEY = st.secrets["MONSTER_API_KEY"]
import time

# --- SETUP ---
SIMILARITY_THRESHOLD = 0.4
MAX_RESULTS = 10
llm_name = "Meta-Llama"
monster_ai_model_name = {
    "Google-Gemma": "google/gemma-2-9b-it",
    "Mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "Microsoft-Phi": "microsoft/Phi-3-mini-4k-instruct",
    "Meta-Llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

# --- LOAD DATA ---
df = pd.read_csv("dataset/dataset_final.csv")
embeddings = np.load("dataset/dataset_final.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- MONSTER API SETUP ---
def setup_monster_api():
    return OpenAI(
        base_url="https://llm.monsterapi.ai/v1/",
        api_key=(MONSTER_API_KEY),
    )

def call_monster_api(user_input, context, client, retries=3, wait_time=5):
    attempt = 0
    while attempt < retries:
        try:
            response = client.chat.completions.create(
                model=monster_ai_model_name[llm_name],
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_input},
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e):
                time.sleep(wait_time)
                attempt += 1
            else:
                break
    return "Could not generate a response."

# --- RECOMMENDATION LOGIC ---
def recommend_tests(user_query):
    query_embedding = model.encode([user_query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    filtered_indices = [idx for idx in sorted_indices if similarities[idx] >= SIMILARITY_THRESHOLD]
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
    return call_monster_api(user_input=message, context=context, client=client)

# --- STREAMLIT UI ---
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🧠 SHL Assessment Recommendation Engine")

query = st.text_area("Enter your hiring or assessment need:", height=100)

if st.button("🔍 Recommend Tests") and query.strip():
    with st.spinner("Running semantic search and generating summary..."):
        results = recommend_tests(query)
        client = setup_monster_api()
        llm_output = generate_llm_response(query, results.head(5), client)

    st.subheader("📝 LLM Summary")
    st.markdown(llm_output)

    st.subheader("📋 Top Recommendations")
    results_display = results[["name", "test_type", "Assessment Length", "url", "similarity"]].copy()
    results_display["url"] = results_display["url"].apply(lambda u: f"https://www.shl.com{u}" if not u.startswith("http") else u)
    st.dataframe(results_display.reset_index(drop=True), use_container_width=True)


    