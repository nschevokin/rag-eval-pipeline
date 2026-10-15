CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE corpora (
    corpus_sha256 text PRIMARY KEY CHECK (corpus_sha256 ~ '^[0-9a-f]{64}$'),
    name text NOT NULL,
    manifest jsonb NOT NULL,
    registered_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE documents (
    doc_sha256 text PRIMARY KEY CHECK (doc_sha256 ~ '^[0-9a-f]{64}$'),
    name text NOT NULL,
    source_url text NOT NULL,
    title text NOT NULL,
    n_bytes integer NOT NULL CHECK (n_bytes > 0),
    ingested_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE corpus_documents (
    corpus_sha256 text NOT NULL REFERENCES corpora,
    doc_sha256 text NOT NULL,
    name text NOT NULL,
    PRIMARY KEY (corpus_sha256, doc_sha256),
    UNIQUE (corpus_sha256, name)
);

CREATE TABLE sections (
    section_id text PRIMARY KEY,
    doc_sha256 text NOT NULL REFERENCES documents,
    parser_version text NOT NULL,
    ord integer NOT NULL CHECK (ord >= 0),
    section_ref text NOT NULL,
    section_refs text[] NOT NULL CHECK (cardinality(section_refs) >= 1),
    heading_path text[] NOT NULL,
    body text NOT NULL,
    UNIQUE (doc_sha256, parser_version, ord)
);

CREATE TABLE chunks (
    chunk_id text PRIMARY KEY,
    chunker_key text NOT NULL,
    doc_sha256 text NOT NULL REFERENCES documents,
    ord integer NOT NULL CHECK (ord >= 0),
    section_ref text NOT NULL,
    section_refs text[] NOT NULL CHECK (cardinality(section_refs) >= 1),
    heading_path text[] NOT NULL,
    n_tokens integer NOT NULL CHECK (n_tokens > 0),
    body text NOT NULL,
    embed_text text NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', embed_text)) STORED,
    UNIQUE (chunker_key, doc_sha256, ord)
);

CREATE INDEX chunks_chunker_doc_idx ON chunks (chunker_key, doc_sha256);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);

CREATE TABLE ingest_receipts (
    doc_sha256 text NOT NULL REFERENCES documents,
    stage text NOT NULL CHECK (stage IN ('sections', 'chunks')),
    stage_key text NOT NULL,
    n_rows integer NOT NULL CHECK (n_rows >= 0),
    completed_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_sha256, stage, stage_key)
);

CREATE TABLE index_versions (
    index_id text PRIMARY KEY CHECK (index_id ~ '^[0-9a-f]{16}$'),
    identity jsonb NOT NULL,
    corpus_sha256 text NOT NULL REFERENCES corpora,
    chunker_key text NOT NULL,
    embed_model text NOT NULL,
    embed_revision text NOT NULL,
    dim integer NOT NULL CHECK (dim = 384),
    params jsonb NOT NULL,
    n_chunks integer NOT NULL CHECK (n_chunks > 0),
    build_seconds double precision NOT NULL CHECK (build_seconds >= 0),
    built_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE embeddings (
    index_id text NOT NULL REFERENCES index_versions,
    chunk_id text NOT NULL REFERENCES chunks,
    embedding vector(384) NOT NULL,
    PRIMARY KEY (index_id, chunk_id)
) PARTITION BY LIST (index_id);

CREATE TABLE embedding_cache (
    model_key text NOT NULL,
    text_sha256 text NOT NULL,
    embedding vector(384) NOT NULL,
    PRIMARY KEY (model_key, text_sha256)
);

CREATE TABLE evaluations (
    eval_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_key text NOT NULL,
    index_id text NOT NULL REFERENCES index_versions,
    retriever_key text NOT NULL,
    retriever jsonb NOT NULL,
    split text NOT NULL CHECK (split IN ('dev', 'test')),
    question_set_sha256 text NOT NULL,
    n_questions integer NOT NULL CHECK (n_questions > 0),
    metrics jsonb NOT NULL,
    timing jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (eval_id, index_id, retriever_key, split)
);

CREATE INDEX evaluations_run_key_idx ON evaluations (run_key);

CREATE TABLE eval_results (
    eval_id bigint NOT NULL REFERENCES evaluations,
    question_id text NOT NULL,
    retrieved text[] NOT NULL,
    metrics jsonb NOT NULL,
    PRIMARY KEY (eval_id, question_id)
);

CREATE TABLE gate_decisions (
    decision_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    eval_id bigint NOT NULL,
    index_id text NOT NULL,
    retriever_key text NOT NULL,
    split text NOT NULL DEFAULT 'dev' CHECK (split = 'dev'),
    passed boolean NOT NULL,
    reasons jsonb NOT NULL,
    floors jsonb NOT NULL,
    regression_tolerance double precision NOT NULL CHECK (regression_tolerance >= 0),
    served_publication_id bigint,
    baseline_eval_id bigint REFERENCES evaluations,
    decided_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (eval_id, index_id, retriever_key, split)
        REFERENCES evaluations (eval_id, index_id, retriever_key, split),
    UNIQUE (decision_id, index_id, retriever_key, passed)
);

CREATE TABLE publications (
    publication_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    decision_id bigint NOT NULL,
    index_id text NOT NULL,
    retriever_key text NOT NULL,
    passed boolean NOT NULL DEFAULT true CHECK (passed),
    retriever jsonb NOT NULL,
    published_by text NOT NULL,
    published_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (decision_id, index_id, retriever_key, passed)
        REFERENCES gate_decisions (decision_id, index_id, retriever_key, passed),
    UNIQUE (decision_id),
    UNIQUE (publication_id, index_id, retriever_key)
);

ALTER TABLE gate_decisions
    ADD FOREIGN KEY (served_publication_id) REFERENCES publications;

CREATE TABLE current_index (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    publication_id bigint NOT NULL,
    index_id text NOT NULL,
    retriever_key text NOT NULL,
    switched_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (publication_id, index_id, retriever_key)
        REFERENCES publications (publication_id, index_id, retriever_key)
);

CREATE TABLE answer_cache (
    input_sha256 text PRIMARY KEY,
    model text NOT NULL,
    response jsonb NOT NULL,
    input_tokens integer NOT NULL CHECK (input_tokens >= 0),
    output_tokens integer NOT NULL CHECK (output_tokens >= 0),
    created_at timestamptz NOT NULL DEFAULT now()
);
