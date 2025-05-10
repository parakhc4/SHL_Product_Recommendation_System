import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import time
import re

MONSTER_API_KEY = st.secrets["MONSTER_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

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
df = pd.read_csv("dataset/dataset_openai.csv")
embeddings = np.load("dataset/dataset_openai.npy")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

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
def get_openai_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def recommend_tests(user_query):
    query_embedding = get_openai_embedding(user_query)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
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

    st.markdown(llm_output)

    st.subheader("📋 Top Recommendations")
    results_display = results[["name", "url", "similarity", "Assessment Length"]].copy()

    results_display["name"] = results_display.apply(
        lambda row: f"[{row['name']}](https://www.shl.com{row['url']})" if not row['url'].startswith("http") else f"[{row['name']}]({row['url']})",
        axis=1
    )
    results_display["Match %"] = results_display["similarity"].apply(lambda x: f"{x * 100:.2f}%")
    results_display["Time Required"] = results_display["Assessment Length"].apply(
        lambda x: (
            f"{match.group()} minutes" if pd.notnull(x) and (match := re.search(r"\d+", str(x))) else "N/A"
        )
    )
    final_df = results_display[["name", "Match %", "Time Required"]]
    st.write(final_df.to_markdown(index=False), unsafe_allow_html=True)