CREATE TABLE IF NOT EXISTS public.raw_datasets (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    source_name text NOT NULL,
    ingested_at timestamptz DEFAULT now() NOT NULL,
    row_count int,
    column_schema jsonb,
    storage_path text NOT NULL
);

-- Index to quickly fetch the latest dataset by source name
CREATE INDEX IF NOT EXISTS idx_raw_datasets_source_ingested 
ON public.raw_datasets (source_name, ingested_at DESC);
