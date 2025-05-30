from fastapi import FastAPI
from pydantic import BaseModel
from recommender.embeddings import load_data, load_model
from recommender.logic import recommend_tests
from recommender.llm import setup_monster_api
import pandas as pd
import re
from dotenv import load_dotenv
import os
print("🔐 OPENAI prefix:", os.getenv("OPENAI_API_KEY")[:5] if os.getenv("OPENAI_API_KEY") else "❌ Not found")
print("Current dir:", os.getcwd())
print("Files in /app:", os.listdir('.'))
print("Files in /app/dataset:", os.listdir('./dataset'))

app = FastAPI(title="SHL Assessment Recommendation API")

df, embeddings = load_data()
model = load_model()
client = setup_monster_api()

class QueryRequest(BaseModel):
    prompt: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommend")
def recommend_endpoint(query: QueryRequest):
    user_input = query.prompt
    results = recommend_tests(user_input, model, df, embeddings)

    formatted_results = []
    for _, row in results.head(10).iterrows():
        formatted_results.append({
            "url": f"https://www.shl.com{row['url']}" if not row['url'].startswith("http") else row['url'],
            "adaptive_support": row["adaptive_support"],
            "remote_support": row["remote_testing"],
            "test_type": [t.strip() for t in row["test_type"].split(",")],
            "duration": int(re.search(r"(\d+)", str(row["Assessment Length"])).group(1)) if pd.notnull(row["Assessment Length"]) else None,
            "description": "Test Descriptions are available on the website."
        })

    return {
        "recommended_assessments": formatted_results
    }
