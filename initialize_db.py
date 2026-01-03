#!/usr/bin/env python3
import duckdb
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_database(db_path: Path, reports_dir: Path, data_dir: Path):
    """Initialize DuckDB and ingest initial metadata."""
    
    # Connect to DuckDB (creates the file if it doesn't exist)
    con = duckdb.connect(str(db_path))
    
    logger.info(f"Connecting to database at {db_path}")

    # 1. document_inventory (Full Metadata Scan)
    inventory_csv = reports_dir / 'pdf_inventory.csv'
    if inventory_csv.exists():
        logger.info(f"Ingesting document inventory from {inventory_csv}...")
        con.execute("DROP TABLE IF EXISTS document_inventory")
        # Use read_csv_auto to handle columns correctly
        con.execute(f"CREATE TABLE document_inventory AS SELECT * FROM read_csv_auto('{inventory_csv}')")
        
        # Add index on md5_hash for fast lookups
        con.execute("CREATE INDEX IF NOT EXISTS idx_md5 ON document_inventory (md5_hash)")
        
        count = con.execute("SELECT COUNT(*) FROM document_inventory").fetchone()[0]
        logger.info(f"Ingested {count} records into document_inventory.")
    else:
        logger.warning(f"Inventory CSV not found: {inventory_csv}")

    # 2. zoning_cases (from official CSV)
    zoning_csv = data_dir / 'gis' / 'Zoning_Cases_20251226.csv'
    if zoning_csv.exists():
        logger.info(f"Ingesting zoning cases from {zoning_csv}...")
        con.execute("DROP TABLE IF EXISTS raw_zoning_cases")
        con.execute(f"CREATE TABLE raw_zoning_cases AS SELECT * FROM read_csv_auto('{zoning_csv}')")
        
        # Create a cleaned/standardized view or table
        # We'll use Case Number as the primary key often
        con.execute("CREATE INDEX IF NOT EXISTS idx_case_num ON raw_zoning_cases (CASE_NUMBER)")
        
        count = con.execute("SELECT COUNT(*) FROM raw_zoning_cases").fetchone()[0]
        logger.info(f"Ingested {count} records into raw_zoning_cases.")
    else:
        logger.warning(f"Zoning CSV not found: {zoning_csv}")

    # 3. case_documents (join table - will be populated during extraction)
    con.execute("""
        CREATE TABLE IF NOT EXISTS case_documents (
            case_number TEXT,
            md5_hash TEXT,
            PRIMARY KEY (case_number, md5_hash)
        )
    """)

    con.close()
    logger.info(f"Database initialization sequence complete.")

if __name__ == "__main__":
    workspace_dir = Path("/Users/chuck/Documents/antigravity_workspace")
    db_path = workspace_dir / "austin_planning.db"
    reports_dir = workspace_dir / "reports"
    data_dir = workspace_dir / "data"
    
    initialize_database(db_path, reports_dir, data_dir)
