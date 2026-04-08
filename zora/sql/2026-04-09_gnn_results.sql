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
