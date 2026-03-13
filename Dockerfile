# ─────────────────────────────────────────────────────────────
# Dockerfile — Enterprise RAG Chatbot
# ─────────────────────────────────────────────────────────────

# Step 1: Start from official Python 3.10 slim image
# "slim" = smaller image, no unnecessary OS packages
FROM python:3.10-slim

# Step 2: Set the working directory inside the container
# All subsequent commands run from /app
WORKDIR /app

# Step 3: Install system dependencies
# libgl1 and libglib2.0-0 are required by some PDF processing libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy requirements.txt first (before the rest of the code)
# WHY: Docker caches each step. If we copy requirements first and
# install them, Docker won't reinstall libraries on every code change —
# only when requirements.txt itself changes. Saves a lot of build time.
COPY requirements.txt .

# Step 5: Install all Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the entire project into the container
COPY . .

# Step 7: Create the data directory if it doesn't exist
# Users will mount their PDFs here or we pre-load sample docs
RUN mkdir -p data/sampledocs vectorstore

# Step 8: Expose port 8501
# This is Streamlit's default port — tells Docker this port is used
EXPOSE 8501

# Step 9: Health check — AWS uses this to know if the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Step 10: Run the app
# --server.address=0.0.0.0 → accept connections from outside the container
# --server.port=8501 → use port 8501
# --server.headless=true → run without opening a browser (server mode)
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
