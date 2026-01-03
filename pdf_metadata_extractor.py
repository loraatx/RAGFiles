#!/usr/bin/env python3
"""
PDF Metadata Extractor for Austin Planning RAG System

This script scans all PDFs in the data directory and extracts:
- File size
- Page count
- Creation/modification dates
- Whether text is extractable (native vs scanned)
- Sample text for classification

Output: CSV file with all metadata for analysis
"""

import os
import csv
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import logging

# Try to import pymupdf (fitz) - the primary PDF library
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("WARNING: PyMuPDF not installed. Install with: pip install pymupdf")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import hashlib

@dataclass
class PDFMetadata:
    """Metadata extracted from a single PDF file."""
    filename: str
    folder: str
    document_id: str
    file_size_bytes: int
    file_size_mb: float
    page_count: int
    is_text_extractable: bool
    text_char_count: int
    sample_text: str
    creation_date: Optional[str]
    modification_date: Optional[str]
    author: Optional[str]
    title: Optional[str]
    producer: Optional[str]
    has_images: bool
    md5_hash: str  # Added for deduplication
    extraction_error: Optional[str]


def calculate_md5(filepath: Path) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def extract_document_id(filename: str) -> str:
    """Extract the numeric ID from filename pattern like '150772_Document.pdf'."""
    base = Path(filename).stem  # Remove .pdf
    if '_Document' in base:
        return base.split('_Document')[0]
    return base


def sanitize_string(s: Optional[str]) -> Optional[str]:
    """Remove surrogates and other problematic characters from strings."""
    if s is None:
        return None
    # Aggressively remove surrogates and re-encode to clean UTF-8
    return s.encode('utf-8', 'ignore').decode('utf-8', 'ignore')


def extract_pdf_metadata(filepath: Path, folder_name: str) -> PDFMetadata:
    """Extract metadata from a single PDF file."""
    filename = filepath.name
    doc_id = extract_document_id(filename)
    file_size = filepath.stat().st_size
    
    # Calculate MD5 early for deduplication
    md5 = calculate_md5(filepath)
    
    metadata = PDFMetadata(
        filename=filename,
        folder=folder_name,
        document_id=doc_id,
        file_size_bytes=file_size,
        file_size_mb=round(file_size / (1024 * 1024), 2),
        page_count=0,
        is_text_extractable=False,
        text_char_count=0,
        sample_text="",
        creation_date=None,
        modification_date=None,
        author=None,
        title=None,
        producer=None,
        has_images=False,
        md5_hash=md5,
        extraction_error=None
    )
    
    if not HAS_PYMUPDF:
        metadata.extraction_error = "PyMuPDF not installed"
        return metadata
    
    try:
        # Optimized: only open if needed
        doc = fitz.open(filepath)
        metadata.page_count = len(doc)
        
        pdf_metadata = doc.metadata
        if pdf_metadata:
            metadata.author = sanitize_string(pdf_metadata.get('author', '')[:100]) or None
            metadata.title = sanitize_string(pdf_metadata.get('title', '')[:200]) or None
            metadata.producer = sanitize_string(pdf_metadata.get('producer', '')[:100]) or None
            metadata.creation_date = sanitize_string(pdf_metadata.get('creationDate', '')[:20]) or None
            metadata.modification_date = sanitize_string(pdf_metadata.get('modDate', '')[:20]) or None
        
        # Sample text assessment
        total_text = ""
        has_images = False
        
        # Faster sampling: check 1st and mid page instead of 10 pages for 27k files
        sample_pages = [0]
        if len(doc) > 1:
            sample_pages.append(len(doc) // 2)
            
        for page_num in sample_pages:
            if page_num >= len(doc): continue
            page = doc[page_num]
            text = page.get_text()
            total_text += text
            if page.get_images():
                has_images = True
        
        metadata.has_images = has_images
        metadata.text_char_count = len(total_text)
        
        avg_chars_per_page = len(total_text) / max(len(sample_pages), 1)
        metadata.is_text_extractable = avg_chars_per_page > 50
        
        # Aggressively sanitize sample text
        sample = sanitize_string(total_text[:500])
        sample = sample.replace('\n', ' ').replace('\r', ' ')
        metadata.sample_text = ' '.join(sample.split())
        
        doc.close()
        
    except Exception as e:
        metadata.extraction_error = sanitize_string(str(e)[:200])
    
    return metadata



def scan_directory(data_dir: Path) -> list[PDFMetadata]:
    """Scan all PDF folders and extract metadata."""
    all_metadata = []
    
    folders = [
        'PlanningCommissionMeetings',
        'BoardOfAdjustmentMeetings',
        'ZoningAndPlattingMeetings'
    ]
    
    for folder in folders:
        folder_path = data_dir / folder
        if not folder_path.exists():
            logger.warning(f"Folder not found: {folder_path}")
            continue
        
        pdf_files = list(folder_path.glob('*.pdf'))
        logger.info(f"Processing {len(pdf_files)} PDFs in {folder}")
        
        for i, pdf_path in enumerate(pdf_files):
            if (i + 1) % 50 == 0:
                logger.info(f"  Progress: {i + 1}/{len(pdf_files)}")
            
            metadata = extract_pdf_metadata(pdf_path, folder)
            all_metadata.append(metadata)
    
    return all_metadata


def generate_report(metadata_list: list[PDFMetadata], output_dir: Path):
    """Generate CSV and summary reports."""
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    csv_path = output_dir / 'pdf_inventory.csv'
    fieldnames = list(asdict(metadata_list[0]).keys())
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for meta in metadata_list:
            writer.writerow(asdict(meta))
    
    logger.info(f"CSV inventory written to: {csv_path}")
    
    # Generate summary statistics
    unique_hashes = set()
    total_unique_size_mb = 0
    
    summary = {
        'total_files': len(metadata_list),
        'unique_files': 0,
        'by_folder': {},
        'text_extractable': 0,
        'scanned_or_image': 0,
        'with_errors': 0,
        'total_size_mb': 0,
        'total_unique_size_mb': 0,
        'total_pages': 0,
        'unique_pages': 0,
        'size_distribution': {
            'small_under_100kb': 0,
            'medium_100kb_2mb': 0,
            'large_2mb_10mb': 0,
            'xlarge_over_10mb': 0
        }
    }
    
    for meta in metadata_list:
        is_unique = meta.md5_hash not in unique_hashes
        if is_unique:
            unique_hashes.add(meta.md5_hash)
            summary['unique_files'] += 1
            summary['total_unique_size_mb'] += meta.file_size_mb
            summary['unique_pages'] += meta.page_count
        
        # By folder
        if meta.folder not in summary['by_folder']:
            summary['by_folder'][meta.folder] = {
                'count': 0,
                'text_extractable': 0,
                'total_pages': 0,
                'total_size_mb': 0
            }
        
        folder_stats = summary['by_folder'][meta.folder]
        folder_stats['count'] += 1
        folder_stats['total_pages'] += meta.page_count
        folder_stats['total_size_mb'] += meta.file_size_mb
        
        if meta.is_text_extractable:
            folder_stats['text_extractable'] += 1
            summary['text_extractable'] += 1
        else:
            summary['scanned_or_image'] += 1
        
        if meta.extraction_error:
            summary['with_errors'] += 1
        
        summary['total_size_mb'] += meta.file_size_mb
        summary['total_pages'] += meta.page_count
        
        # Size distribution
        if meta.file_size_bytes < 100 * 1024:
            summary['size_distribution']['small_under_100kb'] += 1
        elif meta.file_size_bytes < 2 * 1024 * 1024:
            summary['size_distribution']['medium_100kb_2mb'] += 1
        elif meta.file_size_bytes < 10 * 1024 * 1024:
            summary['size_distribution']['large_2mb_10mb'] += 1
        else:
            summary['size_distribution']['xlarge_over_10mb'] += 1
    
    # Round totals
    summary['total_size_mb'] = round(summary['total_size_mb'], 2)
    summary['total_size_gb'] = round(summary['total_size_mb'] / 1024, 2)
    summary['total_unique_size_mb'] = round(summary['total_unique_size_mb'], 2)
    summary['total_unique_size_gb'] = round(summary['total_unique_size_mb'] / 1024, 2)
    
    # Write summary JSON
    summary_path = output_dir / 'pdf_inventory_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary written to: {summary_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("PDF INVENTORY SUMMARY (LARGE SCALE)")
    print("="*60)
    print(f"Total PDFs: {summary['total_files']}")
    print(f"Unique PDFs: {summary['unique_files']} ({100*summary['unique_files']//summary['total_files']}% unique)")
    print(f"Total Size: {summary['total_size_gb']} GB")
    print(f"Unique Size: {summary['total_unique_size_gb']} GB")
    print(f"Total Pages: {summary['total_pages']}")
    print(f"Unique Pages: {summary['unique_pages']}")
    print(f"\nText Extractable: {summary['text_extractable']} ({100*summary['text_extractable']//summary['total_files']}%)")
    print(f"Scanned/Image-based: {summary['scanned_or_image']} ({100*summary['scanned_or_image']//summary['total_files']}%)")
    print(f"Extraction Errors: {summary['with_errors']}")
    print("\nBy Folder:")
    for folder, stats in summary['by_folder'].items():
        print(f"  {folder}:")
        print(f"    Files: {stats['count']}, Pages: {stats['total_pages']}, Size: {round(stats['total_size_mb'], 1)} MB")
        print(f"    Text Extractable: {stats['text_extractable']} ({100*stats['text_extractable']//stats['count']}%)")
    print("\nSize Distribution:")
    for bucket, count in summary['size_distribution'].items():
        print(f"  {bucket}: {count}")
    print("="*60)
    
    return summary


def main():
    """Main entry point."""
    # Determine paths
    script_dir = Path(__file__).parent
    workspace_dir = script_dir.parent
    data_dir = workspace_dir / 'data'
    output_dir = workspace_dir / 'reports'
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return
    
    # Scan and extract
    logger.info("Starting PDF metadata extraction...")
    metadata_list = scan_directory(data_dir)
    
    if not metadata_list:
        print("No PDFs found!")
        return
    
    # Generate reports
    generate_report(metadata_list, output_dir)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
