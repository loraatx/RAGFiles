#!/usr/bin/env python3
"""
FastAPI server for Austin Planning RAG System
Deployed on Google Cloud Run
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import chromadb
from pathlib import Path
import logging
import httpx
import openai

# Import from existing query_rag script
from query_rag import SYSTEM_PROMPT, format_context

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global clients (initialized at startup, reused for all requests)
_openai_client: openai.AsyncOpenAI = None
_chroma_collection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize clients at startup, cleanup at shutdown."""
    global _openai_client, _chroma_collection

    logger.info("Starting up Austin Planning RAG API...")

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    # Initialize async OpenAI client with proper connection pooling
    _openai_client = openai.AsyncOpenAI(
        api_key=api_key,
        timeout=60.0,
        max_retries=3,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30.0
            ),
            timeout=httpx.Timeout(60.0, connect=10.0)
        )
    )

    # Initialize ChromaDB
    chroma_dir = Path("/app/chroma_db")
    if not chroma_dir.exists():
        chroma_dir = Path("./chroma_db")  # Fallback for local dev

    if not chroma_dir.exists():
        logger.warning(f"ChromaDB directory not found at {chroma_dir}")
    else:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        _chroma_collection = client.get_collection(name="austin_planning_docs")
        logger.info(f"ChromaDB loaded: {_chroma_collection.count():,} chunks")

    logger.info("Startup complete!")

    yield  # App runs here

    # Cleanup on shutdown
    logger.info("Shutting down...")
    if _openai_client:
        await _openai_client.close()


# Initialize FastAPI with lifespan
app = FastAPI(
    title="Austin Planning RAG API",
    description="Query Austin planning and zoning documents",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - Allow your GitHub Pages site
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.github.io",  # Your GitHub Pages
        "http://localhost:3000",  # Local development
        "http://localhost:8000",
        "*",  # Allow all for testing (restrict in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_collection():
    """Get ChromaDB collection."""
    if _chroma_collection is None:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    return _chroma_collection


async def get_embedding(text: str) -> list[float]:
    """Generate embedding using async OpenAI client."""
    response = await _openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


async def ask_llm_async(query: str, context: str) -> str:
    """Call OpenAI chat API asynchronously."""
    user_prompt = f"""QUESTION: {query}

RELEVANT DOCUMENTS:
{context}

Based on the documents above, please answer the question. Cite specific case numbers and addresses when mentioned in the sources."""

    response = await _openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=2000,
        temperature=0.3
    )
    return response.choices[0].message.content

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    n_results: int = 10
    show_sources: bool = False

class Source(BaseModel):
    text: str
    case_number: Optional[str] = None
    address: Optional[str] = None
    parcel: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[list[Source]] = None
    query: str

# Health check
@app.get("/")
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Austin Planning RAG API",
        "version": "1.0.0"
    }

# Main query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the Austin Planning RAG system.

    Example:
    ```
    POST /query
    {
        "query": "What are setback requirements for MF-3?",
        "n_results": 10,
        "show_sources": true
    }
    ```
    """
    try:
        logger.info(f"Query received: {request.query}")

        # Get collection
        collection = get_collection()

        # Generate embedding asynchronously (reuses connection pool)
        logger.info("Generating query embedding...")
        query_embedding = await get_embedding(request.query)
        logger.info("Embedding generated successfully")

        # Vector search using embedding directly
        logger.info("Querying ChromaDB...")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.n_results
        )
        logger.info(f"Retrieved {len(results['documents'][0])} results")

        if not results['documents'][0]:
            raise HTTPException(status_code=404, detail="No relevant documents found")

        # Format context
        context = format_context(results)

        # Call LLM asynchronously
        logger.info("Calling LLM...")
        answer = await ask_llm_async(request.query, context)

        # Prepare response
        response = QueryResponse(
            answer=answer,
            query=request.query
        )

        # Add sources if requested
        if request.show_sources:
            sources = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                sources.append(Source(
                    text=doc[:200] + "..." if len(doc) > 200 else doc,
                    case_number=metadata.get('case_0'),
                    address=metadata.get('address_0'),
                    parcel=metadata.get('parcel_0')
                ))
            response.sources = sources

        logger.info("Query completed successfully")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Case lookup endpoint
@app.get("/case/{case_number}")
async def lookup_case(case_number: str):
    """
    Look up information about a specific case.

    Example: GET /case/C14-2009-0151
    """
    try:
        collection = get_collection()

        # Generate embedding for case search
        query_text = f"case {case_number} zoning planning decision"
        query_embedding = await get_embedding(query_text)

        # Search for case using embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where={"case_0": case_number}
        )

        if not results['documents'][0]:
            raise HTTPException(status_code=404, detail=f"Case {case_number} not found")

        # Format context
        context = format_context(results)

        # Call LLM
        query = f"What was the outcome of case {case_number}?"
        answer = await ask_llm_async(query, context)

        return QueryResponse(
            answer=answer,
            query=query
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Case lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Address lookup endpoint
@app.get("/address/{address:path}")
async def lookup_address(address: str):
    """
    Look up zoning/cases for a specific address.

    Example: GET /address/835%20West%206th%20Street
    """
    try:
        collection = get_collection()

        # Generate embedding for address search
        query_text = f"address {address} zoning variance planning"
        query_embedding = await get_embedding(query_text)

        # Search for address using embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            where={"has_addresses": True}
        )

        if not results['documents'][0]:
            raise HTTPException(status_code=404, detail=f"No information found for {address}")

        # Format context
        context = format_context(results)

        # Call LLM
        query = f"What zoning or planning information is available for {address}?"
        answer = await ask_llm_async(query, context)

        return QueryResponse(
            answer=answer,
            query=query
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Address lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Parcel/TCAD ID lookup endpoint
@app.get("/parcel/{tcad_id}")
async def lookup_parcel(tcad_id: str):
    """
    Look up zoning/cases for a specific TCAD parcel ID.

    Example: GET /parcel/123456
    """
    try:
        collection = get_collection()

        # Generate embedding for parcel search
        query_text = f"parcel {tcad_id} zoning planning TCAD"
        query_embedding = await get_embedding(query_text)

        # Search for parcel using embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            where={"has_parcels": True}
        )

        if not results['documents'][0]:
            raise HTTPException(status_code=404, detail=f"No information found for parcel {tcad_id}")

        # Format context
        context = format_context(results)

        # Call LLM
        query = f"What zoning or planning information is available for parcel ID {tcad_id}?"
        answer = await ask_llm_async(query, context)

        return QueryResponse(
            answer=answer,
            query=query
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Parcel lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# For local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
