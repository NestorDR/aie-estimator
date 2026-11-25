-- 0002_quarantine_and_validation_report.sql
-- Add validation_report to ingestion_jobs and idempotent quarantined_records table

ALTER TABLE public.ingestion_jobs
    ADD COLUMN IF NOT EXISTS validation_report jsonb;

COMMENT ON COLUMN public.ingestion_jobs.validation_report IS 'JSON summary of cleaning, deduplication, and validation outcomes';

CREATE TABLE IF NOT EXISTS public.quarantined_records
(
    id              integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id          uuid        NOT NULL REFERENCES public.ingestion_jobs (job_id) ON DELETE CASCADE,
    source_name     text        NOT NULL,
    record_key      text        NOT NULL,
    content_hash    text        NOT NULL,
    failure_reasons jsonb       NOT NULL,
    raw_payload     jsonb       NOT NULL,
    seen_count      integer     NOT NULL DEFAULT 1,
    first_seen_at   timestamptz NOT NULL DEFAULT now(),
    last_seen_at    timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_quarantined_source_key_hash UNIQUE (source_name, record_key, content_hash)
);

COMMENT ON TABLE public.quarantined_records IS 'Idempotent quarantine store for recoverable ingestion validation and duplicate-conflict records';
COMMENT ON COLUMN public.quarantined_records.id IS 'Internal surrogate identifier';
COMMENT ON COLUMN public.quarantined_records.job_id IS 'Most recent ingestion job that observed this record';
COMMENT ON COLUMN public.quarantined_records.source_name IS 'Audited catalog source identifier';
COMMENT ON COLUMN public.quarantined_records.record_key IS 'Deterministic business or blob key identifying the record';
COMMENT ON COLUMN public.quarantined_records.content_hash IS 'SHA-256 digest of the normalized record payload';
COMMENT ON COLUMN public.quarantined_records.failure_reasons IS 'List of validation or conflict reasons routing the record to quarantine';
COMMENT ON COLUMN public.quarantined_records.raw_payload IS 'Original parsed payload preserved for human review';
COMMENT ON COLUMN public.quarantined_records.seen_count IS 'Number of ingestion runs that observed this identical quarantined record';
COMMENT ON COLUMN public.quarantined_records.first_seen_at IS 'Timestamp when the record was first quarantined';
COMMENT ON COLUMN public.quarantined_records.last_seen_at IS 'Timestamp when the record was most recently observed';
