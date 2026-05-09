-- =============================================================================
-- MEGA AI — PostgreSQL Initialization Script
-- =============================================================================
-- This script runs automatically on first container startup.
-- It creates the pgvector extension and all required tables.
-- =============================================================================

-- Enable the pgvector extension for embedding storage
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- Core Tables
-- =============================================================================

-- Jobs table: tracks every query submitted to the system
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    
    -- Status constraint
    CONSTRAINT valid_status CHECK (status IN ('pending', 'running', 'completed', 'failed'))
);

-- Agent logs table: structured logging for every agent event
CREATE TABLE IF NOT EXISTS agent_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    agent_id VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    input_hash VARCHAR(64),
    output_hash VARCHAR(64),
    input_payload JSONB,
    output_payload JSONB,
    latency_ms INTEGER,
    token_count INTEGER,
    policy_violations JSONB DEFAULT '[]'::jsonb,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Event type constraint
    CONSTRAINT valid_event_type CHECK (
        event_type IN (
            'agent_start', 'agent_end', 'tool_call', 'budget_check',
            'policy_violation', 'routing_decision', 'compression_triggered',
            'critique_review', 'synthesis_merge', 'meta_proposal'
        )
    )
);

-- Tool calls table: every tool invocation with full input/output
CREATE TABLE IF NOT EXISTS tool_calls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    agent_id VARCHAR(50) NOT NULL,
    tool_name VARCHAR(50) NOT NULL,
    input JSONB NOT NULL,
    output JSONB NOT NULL,
    latency_ms INTEGER,
    accepted BOOLEAN,
    rejection_reason TEXT,
    retry_number INTEGER NOT NULL DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Retry constraint
    CONSTRAINT valid_retry CHECK (retry_number >= 0 AND retry_number <= 2)
);

-- Evaluation runs table: full reproducibility of every eval run
CREATE TABLE IF NOT EXISTS eval_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    test_cases JSONB NOT NULL,
    summary JSONB NOT NULL,
    diff_from_previous JSONB
);

-- Prompt rewrites table: human-in-the-loop prompt improvements
CREATE TABLE IF NOT EXISTS prompt_rewrites (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_run_id UUID REFERENCES eval_runs(id) ON DELETE CASCADE,
    agent_id VARCHAR(50) NOT NULL,
    dimension VARCHAR(50) NOT NULL,
    original_prompt TEXT NOT NULL,
    proposed_prompt TEXT NOT NULL,
    diff TEXT NOT NULL,
    justification TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    performance_delta JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ,
    reviewed_by VARCHAR(100),
    
    -- Status constraint
    CONSTRAINT valid_rewrite_status CHECK (status IN ('pending', 'approved', 'rejected'))
);

-- RAG documents table: chunked documents with vector embeddings
CREATE TABLE IF NOT EXISTS rag_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    source TEXT,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Indexes for Performance
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_agent_logs_job_id ON agent_logs(job_id);
CREATE INDEX IF NOT EXISTS idx_agent_logs_agent_id ON agent_logs(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_logs_timestamp ON agent_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_tool_calls_job_id ON tool_calls(job_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool_name ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_eval_runs_timestamp ON eval_runs(run_timestamp);
CREATE INDEX IF NOT EXISTS idx_prompt_rewrites_status ON prompt_rewrites(status);

-- HNSW index for fast vector similarity search (pgvector)
CREATE INDEX IF NOT EXISTS idx_rag_embeddings
ON rag_documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- Row Level Security (Optional — enable if multi-tenant)
-- =============================================================================
-- ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE agent_logs ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- Comments for Documentation
-- =============================================================================

COMMENT ON TABLE jobs IS 'Tracks every query submitted to the MEGA AI system';
COMMENT ON TABLE agent_logs IS 'Structured audit log for all agent activities';
COMMENT ON TABLE tool_calls IS 'Every tool invocation with full input/output for debugging';
COMMENT ON TABLE eval_runs IS 'Full reproducibility snapshot of each evaluation run';
COMMENT ON TABLE prompt_rewrites IS 'Human-in-the-loop prompt improvement proposals';
COMMENT ON TABLE rag_documents IS 'Chunked documents with 768-dimension vector embeddings';
