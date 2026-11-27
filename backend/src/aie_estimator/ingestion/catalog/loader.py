# src/aie_estimator/ingestion/catalog/loader.py

# --- Python Standard Library ---
from pathlib import Path
from typing import Final

# --- Third Party Libraries ---
from pydantic import ValidationError
import structlog
import yaml

# --- App modules ---
from aie_estimator.ingestion.catalog.models import DataCatalog

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()


def load_catalog(path: str | Path) -> DataCatalog:
    """
    Parse a YAML catalog file and validate it against ``DataCatalog``.

    :param path: Filesystem path to the YAML catalog file.

    :return: Validated DataCatalog instance.
    :raises FileNotFoundError: If the catalog file does not exist.
    :raises yaml.YAMLError: If the file contains invalid YAML syntax.
    :raises ValidationError: If the parsed payload fails DataCatalog schema validation.
    """
    catalog_path = Path(path)
    try:
        raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
        catalog = DataCatalog.model_validate(raw)
    except (yaml.YAMLError, ValidationError) as e:
        log.warning(
            "catalog_validation_failed",
            path=str(catalog_path),
            error_type=type(e).__name__,
        )
        raise

    log.info(
        "catalog_loaded",
        version=catalog.version,
        source_count=len(catalog.sources),
        included_count=len(catalog.included_sources()),
    )
    return catalog
