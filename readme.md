# SHL Assessment Recommendation System - https://shlxparakh.streamlit.app/

A semantic search engine that recommends relevant SHL assessments based on natural language queries like job descriptions or hiring needs. Built as part of SHL's Generative AI Intern Assignment.

---

## 🚀 Features

- **Natural language input**: Enter free-form queries like “I need a Java developer who can collaborate well.”
- **Semantic search**: Finds relevant SHL assessments using vector similarity and OpenAI embeddings.
- **LLM-generated summaries**: Provides a natural-language summary of the recommended tests.
- **Frontend + API**: Streamlit UI + FastAPI backend, both containerized and independently deployable.
- **Evaluation metrics**: Mean Average Precision@5 and Recall@5 computed over real benchmark queries.

---

## 🛠️ Architecture Overview

| Layer       | Tech |
|-------------|------|
| **Frontend** | Streamlit Cloud |
| **Backend**  | FastAPI (Dockerized, hosted on Azure Container Apps) |
| **Embedding** | OpenAI `text-embedding-ada-002` |
| **Summarization** | Meta-LLaMA via MonsterAPI |
| **Vector Search** | Cosine similarity with precomputed `.npy` embeddings |
| **Deployment** | Azure Container Registry + Streamlit Cloud |

---

## 📦 Tech Stack

- `openai` (for embeddings)
- `fastapi`, `uvicorn` (API)
- `streamlit` (frontend)
- `scikit-learn` (similarity + metrics)
- `pandas`, `numpy`, `tqdm` (data + ETL)
- `selenium`, `bs4` (catalog scraping)
- `docker` (containerization)
- `MonsterAPI` (LLM summarization)

---

<!-- ## 🧪 Evaluation

We used the provided `Test_DATA.csv` benchmark set for accuracy evaluation.

| Metric     | Value |
|------------|-------|
| **MAP@5**  | `0.347` |
| **Recall@5** | `0.411` |

Evaluation pipeline computes per-query average precision and recall by comparing predicted test names to the annotated ground truth. -->

---
