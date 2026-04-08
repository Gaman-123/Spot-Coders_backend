-- Zora Master Architecture Update (Non-destructive)
-- Use this in Supabase SQL Editor to sync your schema without losing any data.

-- 1. Create Raw Datasets table for DE Pipeline Integration
CREATE TABLE IF NOT EXISTS public.raw_datasets (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_name text NOT NULL,
    ingested_at timestamptz DEFAULT now(),
    row_count int,
    column_schema jsonb,
    storage_path text NOT NULL,
    created_at timestamptz DEFAULT now()
);

-- 2. Create GNN Results table for Protein Interaction Analysis
CREATE TABLE IF NOT EXISTS public.gnn_results (
    id bigserial PRIMARY KEY,
    run_id text NOT NULL,
    protein text NOT NULL,
    shap_score float,
    centrality float,
    gnn_embedding float[],
    is_hidden_hub boolean DEFAULT false,
    created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_gnn_results_run_id ON public.gnn_results (run_id);

-- 3. Alter Runs table to support new Pipeline metadata
ALTER TABLE public.runs 
    ADD COLUMN IF NOT EXISTS rows_count int,
    ADD COLUMN IF NOT EXISTS cols_count int,
    ADD COLUMN IF NOT EXISTS schema_json jsonb,
    ADD COLUMN IF NOT EXISTS cleaned_rows int,
    ADD COLUMN IF NOT EXISTS quality_score int,
    ADD COLUMN IF NOT EXISTS cleaning_summary jsonb,
    ADD COLUMN IF NOT EXISTS completed_at timestamptz,
    ADD COLUMN IF NOT EXISTS mode text DEFAULT 'upload';

-- 4. Alter Insights table for expanded reporting
ALTER TABLE public.insights
    ADD COLUMN IF NOT EXISTS protein_summary_json jsonb,
    ADD COLUMN IF NOT EXISTS gnn_summary_json jsonb;
