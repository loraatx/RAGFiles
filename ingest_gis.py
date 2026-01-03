#!/usr/bin/env python3
import duckdb
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ingest_geojson(con, table_name, file_path):
    """Ingest a GeoJSON file into a DuckDB table."""
    logger.info(f"Ingesting {file_path} into {table_name}...")
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return

    # Use ST_Read from the spatial extension
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM ST_Read('{file_path}')")
    
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logger.info(f"Successfully ingested {count} features into {table_name}.")

def main():
    workspace_dir = Path("/Users/chuck/Documents/antigravity_workspace")
    db_path = workspace_dir / "austin_planning.db"
    gis_dir = workspace_dir / "data" / "gis"
    
    # Connect and load spatial extension
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL spatial; LOAD spatial;")
    
    # 1. Imagine Austin Corridors
    corridors_file = gis_dir / "Imagine_Austin_Corridors_20251230.geojson"
    ingest_geojson(con, "gis_corridors", corridors_file)
    
    # 2. Zoning (Small Map Scale)
    zoning_file = gis_dir / "Zoning_(Small_Map_Scale)_20251230.geojson"
    ingest_geojson(con, "gis_zoning", zoning_file)
    
    # 3. Land Use Inventory (Detailed) - Large file (300MB+)
    land_use_file = gis_dir / "Land_Use_Inventory_Detailed_20251230.geojson"
    ingest_geojson(con, "gis_land_use", land_use_file)
    
    con.close()
    logger.info("GIS ingestion complete.")

if __name__ == "__main__":
    main()
