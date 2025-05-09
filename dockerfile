# # ---- Stage 1: Build environment ----
# FROM python:3.11-slim AS base

# # Environment setup
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PIP_NO_CACHE_DIR=1

# # Set work directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     gcc \
#     && rm -rf /var/lib/apt/lists/*

# # Install dependencies
# COPY requirements.txt .
# RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# # Copy app code
# COPY . .

# # Expose port for FastAPI
# EXPOSE 8000

# # Start the app
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential gcc && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y build-essential gcc && \
    apt-get autoremove -y && \
    rm -rf /root/.cache /var/lib/apt/lists/* /tmp/*

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
