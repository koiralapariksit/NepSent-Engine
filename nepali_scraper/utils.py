"""
Utility functions for the Nepali sentiment analysis scraper.
Includes logging setup, data validation, and helper functions.
"""

import logging
import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: Path = None, level: str = "INFO") -> logging.Logger:
    """
    Configure root logger with file and console handlers.
    
    Args:
        log_file: Path to log file (if None, only console output)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured root logger
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


# ============================================================================
# DATA VALIDATION & CLEANING
# ============================================================================

def validate_jsonl_file(filepath: Path) -> Dict[str, int]:
    """
    Validate JSONL file integrity.
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        Dict with validation stats
    """
    stats = {
        "total_lines": 0,
        "valid_lines": 0,
        "invalid_lines": 0,
        "errors": []
    }
    
    if not filepath.exists():
        stats["errors"].append(f"File not found: {filepath}")
        return stats
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stats["total_lines"] += 1
            
            if not line.strip():
                continue
            
            try:
                json.loads(line)
                stats["valid_lines"] += 1
            except json.JSONDecodeError as e:
                stats["invalid_lines"] += 1
                stats["errors"].append(f"Line {line_num}: {str(e)}")
    
    return stats


def deduplicate_jsonl(input_file: Path, output_file: Path, 
                      key_field: str = "text") -> Dict[str, int]:
    """
    Remove duplicate records from JSONL file based on specified field.
    
    Args:
        input_file: Input JSONL file
        output_file: Deduplicated output file
        key_field: Field to check for duplicates
        
    Returns:
        Stats dict with before/after counts
    """
    seen = set()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "input_count": 0,
        "output_count": 0,
        "duplicates_removed": 0
    }
    
    with open(input_file, 'r', encoding='utf-8') as inf:
        with open(output_file, 'w', encoding='utf-8') as outf:
            for line in inf:
                stats["input_count"] += 1
                
                try:
                    record = json.loads(line)
                    key_value = record.get(key_field, "")
                    
                    if key_value and key_value not in seen:
                        seen.add(key_value)
                        outf.write(line)
                        stats["output_count"] += 1
                    else:
                        stats["duplicates_removed"] += 1
                except json.JSONDecodeError:
                    continue
    
    return stats


def generate_domain_balance_report(input_file: Path) -> Dict[str, Dict]:
    """
    Analyze domain and platform distribution in collected data.
    
    Args:
        input_file: JSONL data file
        
    Returns:
        Report dict with distributions
    """
    domains = defaultdict(int)
    platforms = defaultdict(int)
    total = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                domains[record.get('domain', 'unknown')] += 1
                platforms[record.get('platform', 'unknown')] += 1
                total += 1
            except json.JSONDecodeError:
                continue
    
    # Calculate percentages
    domain_report = {}
    for domain, count in domains.items():
        pct = (count / total * 100) if total > 0 else 0
        domain_report[domain] = {
            "count": count,
            "percentage": round(pct, 2)
        }
    
    platform_report = {}
    for platform, count in platforms.items():
        pct = (count / total * 100) if total > 0 else 0
        platform_report[platform] = {
            "count": count,
            "percentage": round(pct, 2)
        }
    
    return {
        "total_records": total,
        "domain_distribution": domain_report,
        "platform_distribution": platform_report,
        "generated_at": datetime.now().isoformat()
    }


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def merge_jsonl_files(input_files: List[Path], output_file: Path) -> Dict[str, int]:
    """
    Merge multiple JSONL files into a single file.
    
    Args:
        input_files: List of input file paths
        output_file: Output merged file
        
    Returns:
        Stats dict
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "input_files": len(input_files),
        "total_records": 0,
        "skipped_records": 0
    }
    
    with open(output_file, 'w', encoding='utf-8') as outf:
        for input_file in input_files:
            if not input_file.exists():
                continue
            
            with open(input_file, 'r', encoding='utf-8') as inf:
                for line in inf:
                    if not line.strip():
                        continue
                    
                    try:
                        record = json.loads(line)
                        outf.write(line)
                        stats["total_records"] += 1
                    except json.JSONDecodeError:
                        stats["skipped_records"] += 1
    
    return stats


def filter_by_domain(input_file: Path, output_file: Path, 
                     domain: str) -> Dict[str, int]:
    """
    Extract records from a specific domain.
    
    Args:
        input_file: Input JSONL file
        output_file: Output file for domain subset
        domain: Domain to filter (e.g., "news_politics")
        
    Returns:
        Stats dict
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "input_count": 0,
        "matched_count": 0
    }
    
    with open(input_file, 'r', encoding='utf-8') as inf:
        with open(output_file, 'w', encoding='utf-8') as outf:
            for line in inf:
                if not line.strip():
                    continue
                
                try:
                    record = json.loads(line)
                    stats["input_count"] += 1
                    
                    if record.get('domain') == domain:
                        outf.write(line)
                        stats["matched_count"] += 1
                except json.JSONDecodeError:
                    continue
    
    return stats


# ============================================================================
# REPORTING
# ============================================================================

def generate_scraping_report(data_dir: Path) -> Dict:
    """
    Generate comprehensive scraping report.
    
    Args:
        data_dir: Directory containing collected data files
        
    Returns:
        Report dict with statistics
    """
    raw_corpus_file = data_dir / "01_raw_corpus.jsonl"
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "data_directory": str(data_dir),
        "files_found": []
    }
    
    # Check for data files
    for file in sorted(data_dir.glob("*.jsonl")):
        file_stats = validate_jsonl_file(file)
        report["files_found"].append({
            "filename": file.name,
            "valid_records": file_stats["valid_lines"],
            "invalid_records": file_stats["invalid_lines"]
        })
    
    # Generate domain balance if raw corpus exists
    if raw_corpus_file.exists():
        balance = generate_domain_balance_report(raw_corpus_file)
        report["domain_balance"] = balance
    
    return report


def print_report(report: Dict, title: str = "Report"):
    """Pretty-print a report dict."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")
    
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for k2, v2 in v.items():
                        print(f"    {k2}: {v2}")
                else:
                    print(f"  {k}: {v}")
            print()
        elif isinstance(value, list):
            print(f"{key}: ({len(value)} items)")
            for item in value:
                print(f"  - {item}")
            print()
        else:
            print(f"{key}: {value}")
    
    print(f"{'='*70}\n")


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def load_environment_variables(env_file: Path = None) -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Path to .env file (if None, uses default locations)
        
    Returns:
        Dict of environment variables
    """
    if env_file is None:
        env_file = Path(".env")
    
    env_vars = {}
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
                        os.environ[key.strip()] = value.strip()
    
    return env_vars


def verify_dependencies() -> Dict[str, bool]:
    """
    Check availability of required Python packages.
    
    Returns:
        Dict with package availability status
    """
    required_packages = {
        'requests': 'HTTP client',
        'bs4': 'BeautifulSoup4 (HTML parsing)',
        'numpy': 'NumPy (numerical operations)',
        'pandas': 'Pandas (data manipulation)',
        'sklearn': 'Scikit-learn (ML utilities)',
    }
    
    status = {}
    for package, description in required_packages.items():
        try:
            __import__(package)
            status[package] = True
        except ImportError:
            status[package] = False
    
    return status
