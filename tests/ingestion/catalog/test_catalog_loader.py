# tests/ingestion/catalog/test_catalog_loader.py

# --- Python Standard Library ---
from pathlib import Path

# --- Third Party Libraries ---
from pydantic import ValidationError
import pytest
import yaml

# --- App modules ---
from aie_estimator.ingestion.catalog.inspect import inspect_root
from aie_estimator.ingestion.catalog.loader import load_catalog
from aie_estimator.settings import Environment, Settings


def test_load_catalog_valid_yaml(tmp_path: Path) -> None:
    """
    GIVEN a valid YAML catalog file on disk
    WHEN load_catalog is called
    THEN it returns a validated DataCatalog with expected sources.
    """
    catalog_file = tmp_path / "catalog.yaml"
    catalog_file.write_text(
        """
version: "1.0.0"
sources:
  - name: historical_budgets
    location: budgets
    format: json
    quality:
      completeness: 4
      consistency: 3
      actuality: 5
      reliability: 4
    sensitivity:
      has_pii: true
      pii_flags: ["EMAIL_ADDRESS", "PERSON"]
      access_level: confidential
    decision: include
""",
        encoding="utf-8",
    )

    catalog = load_catalog(catalog_file)
    assert catalog.version == "1.0.0"
    assert len(catalog.included_sources()) == 1


def test_load_catalog_missing_file(tmp_path: Path) -> None:
    """
    GIVEN a non-existent catalog path
    WHEN load_catalog is called
    THEN FileNotFoundError is raised.
    """
    with pytest.raises(FileNotFoundError):
        load_catalog(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    ("content", "expected_exception"),
    [
        ("version: [unclosed", yaml.YAMLError),
        ("version: '1.0.0'\nsources: []\nextra_key: true\n", ValidationError),
    ],
    ids=["invalid_yaml_syntax", "schema_validation_failure"],
)
def test_load_catalog_invalid_content(
    tmp_path: Path,
    content: str,
    expected_exception: type[Exception],
) -> None:
    """
    GIVEN malformed YAML or schema-invalid catalog content
    WHEN load_catalog is called
    THEN the corresponding YAML or Pydantic validation exception is raised.
    """
    catalog_file = tmp_path / "bad.yaml"
    catalog_file.write_text(content, encoding="utf-8")

    with pytest.raises(expected_exception):
        load_catalog(catalog_file)


def test_repository_catalog_validates_against_seed_corpus() -> None:
    """
    GIVEN the default repository catalog and seed corpus configured in Settings
    WHEN load_catalog and inspect_root are executed
    THEN the catalog validates with include/review/exclude entries and seed facts match disk.
    """
    settings = Settings(app_env=Environment.TEST)
    catalog = load_catalog(settings.catalog_path)
    facts = inspect_root(settings.ingestion_data_root)

    assert len(catalog.sources) == 3
    assert [source.name for source in catalog.included_sources()] == ["historical_budgets"]
    assert sum(entry.file_count for entry in facts) == 10
