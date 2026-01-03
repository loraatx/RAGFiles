#!/usr/bin/env python3
"""
Generate embeddings for document chunks and store in ChromaDB.
Uses OpenAI text-embedding-3-small (1536 dimensions).
"""
import duckdb
import chromadb
from chromadb.utils import embedding_functions
import os
from pathlib import Path
import logging
from tqdm import tqdm
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reports/embeddings.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY not found in environment")
    logger.info("Please set: export OPENAI_API_KEY='your-key-here'")
    exit(1)

def setup_chromadb(persist_dir):
    """Initialize ChromaDB client and collection."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    
    # Create embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    
    # Create or get collection
    try:
        collection = client.get_collection(
            name="austin_planning_docs",
            embedding_function=openai_ef
        )
        logger.info(f"Found existing collection with {collection.count()} docs")
    except:
        collection = client.create_collection(
            name="austin_planning_docs",
            embedding_function=openai_ef,
            metadata={"description": "Austin planning documents with spatial metadata"}
        )
        logger.info("Created new ChromaDB collection")
    
    return client, collection

def get_pending_chunks(db_path, collection, batch_size=100):
    """Get chunks that haven't been embedded yet."""
    con = duckdb.connect(str(db_path))
    
    # Get all chunk IDs
    all_chunks = con.execute("SELECT chunk_id FROM pdf_chunks ORDER BY chunk_id").fetchall()
    all_chunk_ids = {row[0] for row in all_chunks}
    
    # Get already embedded IDs
    try:
        existing_ids = set(collection.get()['ids'])
    except:
        existing_ids = set()
    
    # Find pending
    pending_ids = all_chunk_ids - existing_ids
    
    if not pending_ids:
        logger.info("No pending chunks to embed")
        con.close()
        return []
    
    # Get batch of pending chunks with metadata
    pending_list = list(pending_ids)[:batch_size]
    placeholders = ','.join(['?' for _ in pending_list])
    
    query = f"""
        SELECT chunk_id, chunk_text, cases, addresses, tcad_ids, md5_hash
        FROM pdf_chunks
        WHERE chunk_id IN ({placeholders})
    """
    
    results = con.execute(query, pending_list).fetchall()
    con.close()
    
    return results

def generate_embeddings(db_path, collection, batch_size=100):
    """Generate embeddings in batches."""
    total_embedded = 0
    
    while True:
        # Get pending chunks
        chunks = get_pending_chunks(db_path, collection, batch_size)
        
        if not chunks:
            break
        
        logger.info(f"Processing batch of {len(chunks)} chunks...")
        
        # Prepare data
        ids = []
        documents = []
        metadatas = []
        
        for chunk_id, text, cases, addresses, tcad_ids, md5_hash in chunks:
            ids.append(chunk_id)
            documents.append(text)
            
            # Create metadata
            metadata = {
                "md5_hash": md5_hash,
                "has_cases": bool(cases),
                "has_addresses": bool(addresses),
                "has_parcels": bool(tcad_ids)
            }
            
            # Add first case/address/parcel as filterable fields
            if cases and len(cases) > 0:
                metadata["case_0"] = cases[0]
            if addresses and len(addresses) > 0:
                metadata["address_0"] = addresses[0]
            if tcad_ids and len(tcad_ids) > 0:
                metadata["parcel_0"] = tcad_ids[0]
            
            metadatas.append(metadata)
        
        # Add to ChromaDB (will generate embeddings via API)
        try:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            total_embedded += len(chunks)
            logger.info(f"Embedded {len(chunks)} chunks. Total: {total_embedded}")
            
            # Rate limiting: ~3500 requests/min with tier 1
            # With batch_size=100, that's ~35 batches/min max
            # Sleep 2 seconds between batches to be safe
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Failed to embed batch: {e}")
            # If rate limited, wait longer
            if "rate" in str(e).lower():
                logger.info("Rate limited, waiting 60 seconds...")
                time.sleep(60)
            else:
                raise
    
    return total_embedded

if __name__ == "__main__":
    workspace_dir = Path("/Users/chuck/Documents/antigravity_workspace")
    db_path = workspace_dir / "austin_planning.db"
    chroma_dir = workspace_dir / "chroma_db"
    
    # Setup ChromaDB
    client, collection = setup_chromadb(chroma_dir)
    
    # Generate embeddings
    logger.info("Starting embeddings generation...")
    total = generate_embeddings(db_path, collection, batch_size=100)
    
    # Final stats
    logger.info(f"\n=== EMBEDDINGS COMPLETE ===")
    logger.info(f"Total embedded: {total:,}")
    logger.info(f"Collection size: {collection.count():,}")
    
    # Sample query to verify
    logger.info("\nTesting retrieval...")
    results = collection.query(
        query_texts=["zoning variance for commercial property"],
        n_results=3
    )
    logger.info(f"Sample query returned {len(results['ids'][0])} results")
