# recommender/embeddings.py
import os
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load OpenAI key from env or secrets loader
def load_openai_client():
    load_dotenv()  # ✅ this loads .env into os.environ
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Load catalog and embeddings

def load_data():
    df = pd.read_csv("dataset/dataset_openai.csv")
    embeddings = np.load("dataset/dataset_openai.npy")
    return df, embeddings

def load_model():
    # Returns the embedding caller, since model is OpenAI hosted
    client = load_openai_client()

    def encode(texts):
        if isinstance(texts, str):
            texts = [texts]
        return [client.embeddings.create(model="text-embedding-ada-002", input=t).data[0].embedding for t in texts]

    return encode

def get_similarities(query, embeddings, model_encode):
    query_embedding = model_encode(query)[0]
    return cosine_similarity([query_embedding], embeddings)[0]
