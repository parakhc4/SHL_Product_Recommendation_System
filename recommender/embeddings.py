import os
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

def load_openai_client():
    load_dotenv() 
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_data():
    try:
        print("Loading dataset...")
        df = pd.read_csv("dataset/dataset_openai.csv")
        embeddings = np.load("dataset/dataset_openai.npy")
        print(f"Loaded {len(df)} records and {embeddings.shape} vectors.")
        return df, embeddings
    except Exception as e:
        print("Failed to load dataset:", str(e))

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
