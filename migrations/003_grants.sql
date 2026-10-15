DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_reader') THEN
        CREATE ROLE rag_reader NOLOGIN;
    END IF;
END;
$$;

REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC;

GRANT USAGE ON SCHEMA public TO rag_reader;
GRANT SELECT ON
    corpora, corpus_documents, documents, chunks, index_versions, embeddings,
    publications, current_index, gate_decisions, evaluations
TO rag_reader;
