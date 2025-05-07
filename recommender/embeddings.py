import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import *

def load_data():
    df = pd.read_csv(CSV_PATH)
    embeddings = np.load(NPY_PATH)
    return df, embeddings

def load_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)
