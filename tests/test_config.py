"""
Unit tests for nepali_scraper/config.py
Verifies dynamic path resolution, schema versioning, and environment portability.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nepali_scraper import config


def test_project_root_is_path():
    """Verify PROJECT_ROOT is a Path instance and points to actual repo root."""
    assert isinstance(config.PROJECT_ROOT, Path)
    assert config.PROJECT_ROOT.exists()
    assert (config.PROJECT_ROOT / "nepali_scraper").is_dir()
    assert (config.PROJECT_ROOT / "nepali_scraper" / "config.py").is_file()


def test_directories_resolve_relative_to_root():
    """Verify DATA_DIR, LOGS_DIR, TESTS_DIR resolve relative to PROJECT_ROOT."""
    assert config.DATA_DIR == config.PROJECT_ROOT / "data"
    assert config.LOGS_DIR == config.PROJECT_ROOT / "logs"
    assert config.TESTS_DIR == config.PROJECT_ROOT / "tests"
    assert config.DATA_DIR.exists()
    assert config.LOGS_DIR.exists()


def test_no_hardcoded_drive_letters_in_paths():
    """Verify that configuration defines paths dynamically without hardcoded drive roots."""
    output_files = config.OUTPUT_FILES
    for key, path in output_files.items():
        assert isinstance(path, Path)
        # Must be located inside DATA_DIR
        assert path.parent == config.DATA_DIR


def test_schema_version_and_ordinal_mapping():
    """Verify the active schema is 2.0-ordinal with correct mathematical ordering."""
    assert config.SCHEMA_VERSION == "2.0-ordinal"
    assert config.LABEL_SCHEMA[0] == "Negative"
    assert config.LABEL_SCHEMA[1] == "Neutral"
    assert config.LABEL_SCHEMA[2] == "Positive"
    assert config.LABEL_TO_ID["Negative"] == 0
    assert config.LABEL_TO_ID["Neutral"] == 1
    assert config.LABEL_TO_ID["Positive"] == 2


def test_legacy_schema_preserved():
    """Verify legacy schema remains available for reading historical 1.0 artifacts."""
    assert config.LEGACY_SCHEMA_VERSION == "1.0-nominal"
    assert config.LEGACY_LABEL_SCHEMA[0] == "Negative"
    assert config.LEGACY_LABEL_SCHEMA[1] == "Positive"
    assert config.LEGACY_LABEL_SCHEMA[2] == "Neutral"


def test_import_produces_no_stdout_pollution(capsys):
    """Verify importing or accessing config does not print to stdout."""
    import importlib
    importlib.reload(config)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_canonical_domains_registry():
    """Verify CANONICAL_DOMAINS tuple contains the 4 strict canonical domains."""
    expected = ("news_editorial", "ecommerce_reviews", "social_comments", "general_unclassified")
    assert config.CANONICAL_DOMAINS == expected


def test_init_project_directories_callable():
    """Verify init_project_directories can be called without error."""
    config.init_project_directories()
    assert config.DATA_DIR.is_dir()
    assert config.LOGS_DIR.is_dir()
    assert config.CHECKPOINTS_DIR.is_dir()


