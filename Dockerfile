FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ChromaDB
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install numpy FIRST to prevent chromadb from pulling numpy 2.x
RUN pip install "numpy<2.0.0"

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY query_rag.py .

# Copy ChromaDB database - THIS MUST EXIST IN YOUR BUILD DIRECTORY
# If build fails here, make sure chroma_db/ folder is present
COPY chroma_db/ /app/chroma_db/

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
