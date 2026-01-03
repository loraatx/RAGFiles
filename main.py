#!/usr/bin/env python3
"""
FastAPI server for Austin Planning RAG System
Deployed on Google Cloud Run
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import logging

# Import from existing query_rag script
import sys
sys.path.append('scripts')
from query_rag import SYSTEM_PROMPT, format_context, ask_llm_openai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Austin Planning RAG API",
    description="Query Austin planning and zoning documents",
    version="1.0.0"
)

# CORS - Allow your GitHub Pages site
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.github.io",  # Your GitHub Pages
        "http://localhost:3000",  # Local development
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ChromaDB client (lazy load)
_chroma_collection = None

def get_collection():
    """Get or initialize ChromaDB collection."""
    global _chroma_collection
    
    if _chroma_collection is None:
        logger.info("Initializing ChromaDB...")
        
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        
        # Setup ChromaDB
        chroma_dir = Path("/app/chroma_db")
        if not chroma_dir.exists():
            chroma_dir = Path("./chroma_db")  # Fallback for local dev
        
        client = chromadb.PersistentClient(path=str(chroma_dir))
        
        # Get collection WITHOUT embedding function since we'll generate embeddings manually
        _chroma_collection = client.get_collection(
            name="austin_planning_docs"
        )
        
        logger.info(f"ChromaDB loaded: {_chroma_collection.count():,} chunks")
    
    return _chroma_collection

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
        
        # Generate embedding manually with timeout and connection pool limits
        import openai
        import httpx
        
        # Configure httpx client with connection pool limits
        http_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5
            ),
            timeout=30.0
        )
        
        openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30.0,
            http_client=http_client
        )
        
        logger.info("Generating query embedding...")
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=request.query
        )
        query_embedding = embedding_response.data[0].embedding
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
        
        # Call LLM (using OpenAI for Cloud Run)
        logger.info("Calling LLM...")
        answer = ask_llm_openai(request.query, context)
        
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
                    text=doc[:200] + "...",  # First 200 chars
                    case_number=metadata.get('case_0'),
                    address=metadata.get('address_0'),
                    parcel=metadata.get('parcel_0')
                ))
            response.sources = sources
        
        logger.info("Query completed successfully")
        return response
        
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
        
        # Search for case
        results = collection.query(
            query_texts=[f"case {case_number}"],
            n_results=5,
            where={"case_0": {"$contains": case_number}}
        )
        
        if not results['documents'][0]:
            raise HTTPException(status_code=404, detail=f"Case {case_number} not found")
        
        # Format context
        context = format_context(results)
        
        # Call LLM
        query = f"What was the outcome of case {case_number}?"
        answer = ask_llm_openai(query, context)
        
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
        
        # Search for address
        results = collection.query(
            query_texts=[f"address {address} zoning variance"],
            n_results=10,
            where={"has_addresses": True}
        )
        
        if not results['documents'][0]:
            raise HTTPException(status_code=404, detail=f"No information found for {address}")
        
        # Format context
        context = format_context(results)
        
        # Call LLM
        query = f"What zoning or planning information is available for {address}?"
        answer = ask_llm_openai(query, context)
        
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
        
        # Search for parcel
        results = collection.query(
            query_texts=[f"parcel {tcad_id} zoning planning"],
            n_results=10,
            where={"has_parcels": True}
        )
        
        if not results['documents'][0]:
            raise HTTPException(status_code=404, detail=f"No information found for parcel {tcad_id}")
        
        # Format context
        context = format_context(results)
        
        # Call LLM
        query = f"What zoning or planning information is available for parcel ID {tcad_id}?"
        answer = ask_llm_openai(query, context)
        
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
