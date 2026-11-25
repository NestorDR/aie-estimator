# src/aie_estimator/ingestion/pii/pseudonymizer.py

"""
Consistent reversible pseudonymizer ensuring the same input maps to the same pseudonym corpus-wide.

Processing invariants:
1. Offload CPU-bound PII span detection via ``asyncio.to_thread`` (the ``PiiAnalyzer`` contract
   guarantees non-overlapping spans with intersecting ``[start, end)`` matches already resolved).
2. Compute a salted HMAC-SHA256 digest of each original value locally so plaintext PII never
   enters the mapping store or logs.
3. Resolve or lazily generate a type-appropriate Faker pseudonym through the async mapping store
   to preserve semantic co-reference across documents and chunks.
4. Apply replacements right-to-left so earlier character offsets remain valid, recursively scrub
   string leaves in ``DocumentMetadata.extra``, and recompute ``Document.content_hash`` on the
   pseudonymized text for downstream incremental indexing.

Reversibility lives solely in the mapping store: erasing ``(entity_type, original_hash)``
invalidates future lookups for GDPR Article 17 without rewriting already-ingested documents.
"""

# --- Python Standard Library ---
import asyncio
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import hmac
from typing import Any, Final

# --- Third Party Libraries ---
from faker import Faker
from pydantic import SecretStr
import structlog

# --- App modules ---
from aie_estimator.ingestion.catalog.models import PiiFlag
from aie_estimator.ingestion.documents.models import Document, compute_content_hash
from aie_estimator.ingestion.pii.analyzer import PiiAnalyzer
from aie_estimator.ingestion.pii.mapping_store import AsyncMappingStore
from aie_estimator.settings import Settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger(__name__)

# Technical string metadata keys that never carry free-text PII and must remain untouched.
_SKIP_METADATA_KEYS: Final[frozenset[str]] = frozenset({"format_mode", "timestamp", "currency", "signed_at"})


def _unwrap_salt(salt: SecretStr | str | None, *, error_message: str) -> str:
    """
    Extract and validate a non-empty HMAC salt string.

    :param salt: Secret HMAC salt as ``SecretStr``, plain string, or None.
    :param error_message: Error message raised when the salt is missing or blank.

    :return: Validated non-empty salt string.
    :raises ValueError: If ``salt`` is None, empty, or whitespace-only.
    """
    raw_salt = salt.get_secret_value() if isinstance(salt, SecretStr) else (salt or "")
    if not raw_salt.strip():
        raise ValueError(error_message)
    return raw_salt


def compute_pii_hash(original_value: str, salt: SecretStr | str) -> str:
    """
    Compute the salted HMAC-SHA256 hex digest for a plaintext PII value.

    :param original_value: Raw detected entity value.
    :param salt: Secret HMAC salt as ``SecretStr`` or plain string.

    :return: Lowercase 64-character hex digest.
    :raises ValueError: If ``salt`` is empty or whitespace-only.
    """
    raw_salt = _unwrap_salt(salt, error_message="PII HMAC salt must be a non-empty secret.")
    return hmac.new(raw_salt.encode("utf-8"), original_value.encode("utf-8"), hashlib.sha256).hexdigest()


def resolve_pii_salt(settings: Settings) -> SecretStr:
    """
    Resolve and validate the configured HMAC salt from ``Settings``.

    :param settings: Active application Settings instance.

    :return: Validated non-empty ``SecretStr`` HMAC salt.
    :raises ValueError: If ``settings.pii_hash_salt`` is unset or blank.
    """
    _unwrap_salt(
        settings.pii_hash_salt,
        error_message="PII_HASH_SALT must be configured when PII pseudonymization is enabled.",
    )
    # _unwrap_salt already rejected None/blank; assert narrows Optional[SecretStr] for type checkers.
    assert settings.pii_hash_salt is not None  # noqa: S101
    return settings.pii_hash_salt


@dataclass(frozen=True, slots=True)
class AppliedMapping:
    """Immutable audit record of a single entity replacement applied to text."""

    entity_type: str
    original_hash: str
    pseudonym: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class PseudonymizationResult:
    """Immutable outcome of pseudonymizing a single text string."""

    pseudonymized_text: str
    applied: tuple[AppliedMapping, ...] = ()


class ConsistentPseudonymizer:
    """Entity-consistent reversible pseudonymizer backed by an ``AsyncMappingStore``."""

    def __init__(
        self,
        *,
        analyzer: PiiAnalyzer,
        mapping_store: AsyncMappingStore,
        salt: SecretStr | str,
        faker_locale: str = "en_US",
    ) -> None:
        """
        Initialize the pseudonymizer with an analyzer, async mapping store, and HMAC salt.

        :param analyzer: ``PiiAnalyzer`` detecting non-overlapping typed character spans.
        :param mapping_store: ``AsyncMappingStore`` persisting hashed mappings.
        :param salt: Secret HMAC salt used to hash plaintext entities before lookup.
        :param faker_locale: Locale passed to ``Faker`` for synthetic entity generation.
        :raises ValueError: If ``salt`` is empty or whitespace-only.
        """
        self._analyzer = analyzer
        self._store = mapping_store
        self._salt_bytes = _unwrap_salt(salt, error_message="PII HMAC salt must be a non-empty secret.").encode("utf-8")
        self._faker = Faker(faker_locale)
        # Zero-arg factories are evaluated lazily by the mapping store only on cache misses.
        self._generators: dict[str, Callable[[], str]] = {
            PiiFlag.PERSON.value: self._faker.name,
            PiiFlag.EMAIL_ADDRESS.value: self._faker.ascii_company_email,
            PiiFlag.PHONE_NUMBER.value: self._faker.phone_number,
            PiiFlag.LOCATION.value: self._faker.city,
            PiiFlag.ORGANIZATION.value: self._faker.company,
            PiiFlag.IBAN_CODE.value: self._faker.iban,
            PiiFlag.CREDIT_CARD.value: self._faker.credit_card_number,
            PiiFlag.BUDGET_ID.value: lambda: (
                f"BUDGET-{self._faker.random_int(min=2020, max=2029)}-{self._faker.random_int(min=0, max=9999):04d}"
            ),
            PiiFlag.CLIENT_CODE.value: lambda: f"CLI-{self._faker.random_int(min=0, max=9999):04d}",
            PiiFlag.NUMERIC_IDENTIFIER.value: lambda: str(self._faker.random_number(digits=10, fix_len=True)),
        }

    def _fallback_pseudonym(self) -> str:
        return f"REDACTED-{self._faker.uuid4()[:8]}"

    async def pseudonymize_text(
        self,
        text: str,
        *,
        source_name: str | None = None,
    ) -> PseudonymizationResult:
        """
        Detect and consistently pseudonymize all PII spans in ``text`` right-to-left.

        :param text: Raw input text to pseudonymize.
        :param source_name: Optional catalog source identifier recorded on mappings.

        :return: PseudonymizationResult containing safe text and applied mappings.
        """
        if not text:
            return PseudonymizationResult(pseudonymized_text=text, applied=())

        # Offload CPU-bound spaCy/Presidio inference so the async FastAPI event loop stays responsive.
        spans = await asyncio.to_thread(self._analyzer.analyze, text)
        if not spans:
            return PseudonymizationResult(pseudonymized_text=text, applied=())

        # Replace right-to-left: slicing `out` in descending end-offset order keeps all earlier
        # offsets valid even when replacement pseudonyms differ in length from the original text.
        ordered = sorted(spans, key=lambda span: span.end, reverse=True)

        out = text
        applied: list[AppliedMapping] = []
        for span in ordered:
            # Always slice the immutable original `text` (never mutated `out`) before HMAC hashing.
            original_value = text[span.start : span.end]
            original_hash = hmac.new(self._salt_bytes, original_value.encode("utf-8"), hashlib.sha256).hexdigest()
            factory = self._generators.get(span.entity_type, self._fallback_pseudonym)
            pseudonym = await self._store.lookup_or_create(
                entity_type=span.entity_type,
                original_hash=original_hash,
                new_pseudonym_factory=factory,
                source_name=source_name,
            )
            out = out[: span.start] + pseudonym + out[span.end :]
            applied.append(
                AppliedMapping(
                    entity_type=span.entity_type,
                    original_hash=original_hash,
                    pseudonym=pseudonym,
                    start=span.start,
                    end=span.end,
                )
            )

        return PseudonymizationResult(pseudonymized_text=out, applied=tuple(applied))

    async def _pseudonymize_metadata_value(
        self,
        value: Any,
        *,
        source_name: str | None,
        applied: list[AppliedMapping],
    ) -> Any:
        """
        Recursively pseudonymize string leaves inside ``DocumentMetadata.extra``.

        :param value: Arbitrary metadata value (string, list, dict, or scalar).
        :param source_name: Optional catalog source identifier.
        :param applied: Accumulator list collecting AppliedMapping entries.

        :return: Pseudonymized copy of ``value``.
        """
        # Only free-text string leaves can contain PII; numeric/boolean scalars and technical keys pass through untouched.
        if isinstance(value, str):
            result = await self.pseudonymize_text(value, source_name=source_name)
            applied.extend(result.applied)
            return result.pseudonymized_text
        if isinstance(value, list):
            return [await self._pseudonymize_metadata_value(item, source_name=source_name, applied=applied) for item in value]
        if isinstance(value, dict):
            return {
                key: inner
                if key in _SKIP_METADATA_KEYS
                else await self._pseudonymize_metadata_value(inner, source_name=source_name, applied=applied)
                for key, inner in value.items()
            }
        return value

    async def pseudonymize_document(
        self,
        document: Document,
        *,
        source_name: str | None = None,
    ) -> tuple[Document, list[AppliedMapping]]:
        """
        Pseudonymize a canonical ``Document`` (text and ``metadata.extra``) and refresh ``content_hash``.

        :param document: Canonical Document emitted by ``normalize_documents``.
        :param source_name: Optional catalog source override; defaults to ``document.metadata.source_name``.

        :return: Tuple of ``(safe_document, applied_mappings)``.
        """
        effective_source = source_name or document.metadata.source_name
        text_result = await self.pseudonymize_text(document.text, source_name=effective_source)
        applied = list(text_result.applied)

        safe_extra = await self._pseudonymize_metadata_value(
            document.metadata.extra,
            source_name=effective_source,
            applied=applied,
        )
        safe_metadata = document.metadata.model_copy(update={"extra": safe_extra})
        # Recompute content_hash on the safe text so downstream incremental indexing never fingerprints raw PII.
        safe_hash = (
            compute_content_hash(text_result.pseudonymized_text)
            if text_result.pseudonymized_text != document.text
            else document.content_hash
        )
        safe_doc = document.model_copy(
            update={
                "text": text_result.pseudonymized_text,
                "content_hash": safe_hash,
                "metadata": safe_metadata,
            }
        )
        return safe_doc, applied

    async def pseudonymize_documents(
        self,
        documents: list[Document],
        *,
        source_name: str | None = None,
    ) -> list[Document]:
        """
        Pseudonymize a batch of canonical documents and emit a homogeneous completion log.

        :param documents: List of canonical Document instances.
        :param source_name: Optional catalog source identifier for logging and provenance.

        :return: List of pseudonymized Document instances.
        """
        safe_documents: list[Document] = []
        counts_by_type: Counter[str] = Counter()
        for document in documents:
            safe_doc, applied = await self.pseudonymize_document(document, source_name=source_name)
            safe_documents.append(safe_doc)
            for item in applied:
                counts_by_type[item.entity_type] += 1

        log.info(
            "ingestion_pii_completed",
            source_name=source_name,
            documents_count=len(safe_documents),
            entities_replaced_count=sum(counts_by_type.values()),
            entities_by_type=dict(counts_by_type),
        )
        return safe_documents
