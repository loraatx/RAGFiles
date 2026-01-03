#!/usr/bin/env python3
"""
Create chunks from extracted PDF text with metadata injection.
Chunks are fixed-size (1000 tokens) with 200-token overlap.
"""
import duckdb
import tiktoken
from pathlib import Path
import logging
from tqdm import tqdm
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reports/chunking.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Chunking parameters
CHUNK_SIZE = 1000  # tokens
OVERLAP = 200      # tokens
MAX_CHUNKS_PER_DOC = 50  # Limit for very large documents

# Initialize tokenizer (same as OpenAI's)
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    """Count tokens in text using tiktoken."""
    return len(encoding.encode(text))

def create_metadata_prefix(md5, cases, addresses, tcad_ids, chunk_idx):
    """Create metadata prefix for chunk."""
    parts = []
    
    if cases:
        parts.append(f"Cases: {', '.join(cases[:3])}")
    if addresses:
        parts.append(f"Addresses: {', '.join(addresses[:3])}")
    if tcad_ids:
        parts.append(f"Parcels: {', '.join(tcad_ids[:3])}")
    
    parts.append(f"Chunk: {chunk_idx + 1}")
    
    if parts:
        return " | ".join(parts) + "\n---\n"
    return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Split text into overlapping chunks of specified token size.
    Returns list of chunk texts.
    """
    tokens = encoding.encode(text)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move start forward by (chunk_size - overlap)
        start += (chunk_size - overlap)
        
        # Safety limit
        if len(chunks) >= MAX_CHUNKS_PER_DOC:
            logger.warning(f"Hit max chunks limit ({MAX_CHUNKS_PER_DOC})")
            break
    
    return chunks

def setup_chunks_table(con):
    """Create the pdf_chunks table."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS pdf_chunks (
            chunk_id TEXT PRIMARY KEY,
            md5_hash TEXT,
            chunk_index INTEGER,
            chunk_text TEXT,
            chunk_tokens INTEGER,
            cases TEXT[],
            addresses TEXT[],
            tcad_ids TEXT[]
        )
    """)
    logger.info("Chunks table ready")

def process_documents(db_path, batch_size=100):
    """Process documents and create chunks."""
    con = duckdb.connect(str(db_path))
    setup_chunks_table(con)
    
    # Get documents not yet chunked
    query = """
        SELECT t.md5_hash, t.full_text, t.extracted_cases, t.extracted_addresses, t.extracted_tcad_ids
        FROM document_text t
        LEFT JOIN (SELECT DISTINCT md5_hash FROM pdf_chunks) c ON t.md5_hash = c.md5_hash
        WHERE c.md5_hash IS NULL
        ORDER BY length(t.full_text) ASC  -- Process smaller docs first
        LIMIT ?
    """
    
    pending = con.execute(query, [batch_size]).fetchall()
    
    if not pending:
        logger.info("No pending documents to chunk")
        con.close()
        return 0
    
    logger.info(f"Processing {len(pending)} documents...")
    
    total_chunks = 0
    for md5, full_text, cases, addresses, tcad_ids in tqdm(pending):
        try:
            # Skip if text is too short
            if len(full_text) < 100:
                logger.warning(f"Skipping {md5[:8]} - text too short")
                continue
            
            # Create chunks
            text_chunks = chunk_text(full_text)
            
            # Store each chunk
            for idx, text_chunk in enumerate(text_chunks):
                # Add metadata prefix
                prefix = create_metadata_prefix(md5, cases, addresses, tcad_ids, idx)
                chunk_with_metadata = prefix + text_chunk
                
                # Create chunk ID
                chunk_id = f"{md5}_{idx:03d}"
                
                # Count tokens
                tokens = count_tokens(chunk_with_metadata)
                
                # Insert
                con.execute("""
                    INSERT OR IGNORE INTO pdf_chunks 
                    (chunk_id, md5_hash, chunk_index, chunk_text, chunk_tokens, cases, addresses, tcad_ids)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (chunk_id, md5, idx, chunk_with_metadata, tokens, cases, addresses, tcad_ids))
                
                total_chunks += 1
        
        except Exception as e:
            logger.error(f"Failed to chunk {md5[:8]}: {e}")
    
    con.close()
    logger.info(f"Created {total_chunks} chunks from {len(pending)} documents")
    return len(pending)

if __name__ == "__main__":
    workspace_dir = Path("/Users/chuck/Documents/antigravity_workspace")
    db_path = workspace_dir / "austin_planning.db"
    
    total = 0
    while True:
        count = process_documents(db_path, batch_size=500)
        total += count
        if count == 0:
            break
        logger.info(f"Total documents chunked: {total}")
    
    # Final stats
    con = duckdb.connect(str(db_path))
    chunk_count = con.execute("SELECT COUNT(*) FROM pdf_chunks").fetchone()[0]
    doc_count = con.execute("SELECT COUNT(DISTINCT md5_hash) FROM pdf_chunks").fetchone()[0]
    avg_chunks = chunk_count / doc_count if doc_count > 0 else 0
    
    logger.info(f"\n=== CHUNKING COMPLETE ===")
    logger.info(f"Documents: {doc_count:,}")
    logger.info(f"Total chunks: {chunk_count:,}")
    logger.info(f"Avg chunks/doc: {avg_chunks:.1f}")
    con.close()
