#!/usr/bin/env python3
import duckdb
import pypdfium2 as pdfium
import pytesseract
from PIL import Image
import re
from pathlib import Path
import logging
from tqdm import tqdm
import io

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reports/ocr_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Patterns (Shared logic from extract_text.py)
CASE_PATTERNS = [
    r'[A-Z]{1,4}-\d{4}-\d{4}[A-Z]{0,5}',
    r'NPA-\d{4}-\d{4}\.\d{2}',
    r'SP-\d{4}-\d{4}[A-Z]?'
]
ADDRESS_PATTERN = r'\b\d{1,5}\s(?:[NSEW]\.?\s|NORTH\s|SOUTH\s|EAST\s|WEST\s)?[\w\s]{1,30}(?:RD|AVE|ST|DR|BLVD|PL|LN|CT|PKWY|HWY|TRL|CIR|CV|TER|ROAD|AVENUE|STREET|DRIVE|BOULEVARD|PLACE|LANE|COURT|PARKWAY|HIGHWAY|TRAIL|CIRCLE|COVE|TERRACE)\b'
TCAD_PATTERN = r'\b\d{10}\b'

ADDRESS_BLACKLIST = {'301 W 2ND ST', '301 WEST 2ND ST', '301 W SECOND ST', '505 BARTON SPRINGS RD', '6310 WILHELMINA DELCO DR', '100 CONGRESS AVE'}
NOISE_KEYWORDS = {'FEET', 'SUBJECT', 'ITEMS', 'CASES', 'SUBCHAPTER', 'SUBSTANDARD', 'DISTRICT', 'COMBINING', 'SECTION', 'LOT', 'BLOCK', 'PROJECT', 'REQUEST', 'SUBDIVISION'}

def normalize_address(addr):
    if not addr: return ""
    clean = re.sub(r'[.,]', '', addr.upper())
    clean = clean.replace('STREET', 'ST').replace('ROAD', 'RD').replace('DRIVE', 'DR')
    return ' '.join(clean.split())

def extract_entities(text):
    results = {'cases': set(), 'addresses': set(), 'tcad_ids': set()}
    for pattern in CASE_PATTERNS:
        for match in re.findall(pattern, text):
            results['cases'].add(match.strip('.'))
    for match in re.findall(ADDRESS_PATTERN, text, re.IGNORECASE):
        norm_addr = normalize_address(match)
        if norm_addr not in ADDRESS_BLACKLIST and len(norm_addr) > 8:
            if not any(noise in norm_addr for noise in NOISE_KEYWORDS):
                results['addresses'].add(norm_addr)
    for match in re.findall(TCAD_PATTERN, text):
        results['tcad_ids'].add(match)
    return {k: list(v) for k, v in results.items()}

def process_ocr_batch(db_path, data_dir, batch_size=50):
    con = duckdb.connect(str(db_path))
    
    # Get scanned files not yet in document_text
    query = """
        SELECT md5_hash, ANY_VALUE(folder), ANY_VALUE(filename)
        FROM document_inventory i
        LEFT JOIN document_text t USING (md5_hash)
        WHERE i.is_text_extractable = FALSE 
        AND t.md5_hash IS NULL
        GROUP BY md5_hash
        ORDER BY i.file_size_bytes ASC -- Process smaller files first
        LIMIT ?
    """
    
    pending_files = con.execute(query, [batch_size]).fetchall()
    
    if not pending_files:
        logger.info("No scanned PDFs to process.")
        con.close()
        return 0
    
    logger.info(f"Starting OCR batch for {len(pending_files)} scanned PDFs...")
    
    success_count = 0
    for md5, folder, filename in tqdm(pending_files):
        pdf_path = data_dir / folder / filename
        try:
            # Load PDF
            pdf = pdfium.PdfDocument(pdf_path)
            full_text = ""
            
            # Limit OCR to first 10 pages for speed/relevance in planning docs
            pages_to_ocr = min(len(pdf), 10)
            
            for i in range(pages_to_ocr):
                page = pdf[i]
                bitmap = page.render(scale=2) # 144 DPI
                pil_image = bitmap.to_pil()
                
                # Perform OCR
                page_text = pytesseract.image_to_string(pil_image)
                full_text += page_text + "\n"
            
            entities = extract_entities(full_text)
            
            # Store results
            con.execute(
                "INSERT OR IGNORE INTO document_text (md5_hash, full_text, extracted_cases, extracted_addresses, extracted_tcad_ids, extraction_method) VALUES (?, ?, ?, ?, ?, ?)",
                (md5, full_text, entities['cases'], entities['addresses'], entities['tcad_ids'], "Tesseract-OCR")
            )
            
            for case in entities['cases']:
                con.execute("INSERT OR IGNORE INTO case_documents (case_number, md5_hash) VALUES (?, ?)", (case, md5))
            for address in entities['addresses']:
                con.execute("INSERT OR IGNORE INTO document_addresses (md5_hash, address) VALUES (?, ?)", (md5, address))
            for tcad_id in entities['tcad_ids']:
                con.execute("INSERT OR IGNORE INTO document_parcels (md5_hash, tcad_id) VALUES (?, ?)", (md5, tcad_id))
            
            success_count += 1
            pdf.close()
            
        except Exception as e:
            logger.error(f"OCR failed for {filename}: {e}")
            
    con.close()
    return success_count

if __name__ == "__main__":
    workspace_dir = Path("/Users/chuck/Documents/antigravity_workspace")
    db_path = workspace_dir / "austin_planning.db"
    data_dir = workspace_dir / "data"
    
    total = 0
    while True:
        count = process_ocr_batch(db_path, data_dir, batch_size=20)
        total += count
        if count == 0: break
        logger.info(f"Total OCR processed so far: {total}")
