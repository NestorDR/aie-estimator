-- 0001_initial_schema.sql
-- Session 6 initial PostgreSQL schema: ingestion_jobs + pseudonym_mappings

CREATE TABLE IF NOT EXISTS public.ingestion_jobs
(
    job_id          uuid        PRIMARY KEY,
    source_name     text        NOT NULL,
    status          text        NOT NULL DEFAULT 'pending',
    documents_count integer     NOT NULL DEFAULT 0,
    error_message   text,
    started_at      timestamptz NOT NULL DEFAULT now(),
    finished_at     timestamptz
);

COMMENT ON TABLE public.ingestion_jobs IS 'Tracking records for asynchronous offline ingestion runs';
COMMENT ON COLUMN public.ingestion_jobs.job_id IS 'Unique identifier of the ingestion run';
COMMENT ON COLUMN public.ingestion_jobs.source_name IS 'Audited catalog source identifier';
COMMENT ON COLUMN public.ingestion_jobs.status IS 'Lifecycle status: pending, running, completed, failed';
COMMENT ON COLUMN public.ingestion_jobs.documents_count IS 'Total canonical documents emitted by the run';
COMMENT ON COLUMN public.ingestion_jobs.error_message IS 'Truncated failure reason when status is failed';
COMMENT ON COLUMN public.ingestion_jobs.started_at IS 'Timestamp when the ingestion job was created';
COMMENT ON COLUMN public.ingestion_jobs.finished_at IS 'Timestamp when the ingestion job reached a terminal state';

CREATE TABLE IF NOT EXISTS public.pseudonym_mappings
(
    id            integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    entity_type   text        NOT NULL,
    original_hash text        NOT NULL,
    pseudonym     text        NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_mappings_entity_hash UNIQUE (entity_type, original_hash)
);

COMMENT ON TABLE public.pseudonym_mappings IS 'GDPR-auditable hashed mapping store for reversible PII pseudonymization';
COMMENT ON COLUMN public.pseudonym_mappings.id IS 'Internal surrogate identifier';
COMMENT ON COLUMN public.pseudonym_mappings.entity_type IS 'Detected PII entity category (e.g., PERSON, EMAIL, PHONE)';
COMMENT ON COLUMN public.pseudonym_mappings.original_hash IS 'Salted HMAC-SHA256 digest of the original PII value';
COMMENT ON COLUMN public.pseudonym_mappings.pseudonym IS 'Deterministic replacement token emitted into normalized documents';
COMMENT ON COLUMN public.pseudonym_mappings.created_at IS 'Timestamp when the pseudonym mapping was persisted';
