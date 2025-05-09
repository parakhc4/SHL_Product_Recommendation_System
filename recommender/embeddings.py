from openai import OpenAI
import pandas as pd
import numpy as np

def load_model():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def encode(texts):
        if isinstance(texts, str):
            texts = [texts]
        return [client.embeddings.create(model="text-embedding-ada-002", input=t).data[0].embedding for t in texts]
    
    return encode

def load_data():
    df = pd.read_csv("dataset/dataset_openai.csv")
    embeddings = np.load("dataset/dataset_openai.npy")
    return df, embeddings