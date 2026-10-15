CREATE FUNCTION forbid_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% on % is not allowed: rows are append-only', TG_OP, TG_TABLE_NAME
        USING ERRCODE = 'integrity_constraint_violation';
END;
$$;

DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'corpora', 'corpus_documents', 'documents', 'sections', 'chunks', 'ingest_receipts',
        'index_versions', 'evaluations', 'eval_results', 'gate_decisions', 'publications'
    ]
    LOOP
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION forbid_mutation()',
            t || '_append_only', t);
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE TRUNCATE ON %I FOR EACH STATEMENT EXECUTE FUNCTION forbid_mutation()',
            t || '_no_truncate', t);
    END LOOP;
END;
$$;

CREATE FUNCTION current_index_requires_passed_gate() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    ok boolean;
BEGIN
    SELECT d.passed INTO ok
    FROM publications p
    JOIN gate_decisions d ON d.decision_id = p.decision_id
    WHERE p.publication_id = NEW.publication_id;
    IF ok IS DISTINCT FROM true THEN
        RAISE EXCEPTION 'publication % has no passing gate decision', NEW.publication_id
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER current_index_guard
    BEFORE INSERT OR UPDATE ON current_index
    FOR EACH ROW EXECUTE FUNCTION current_index_requires_passed_gate();

CREATE TRIGGER current_index_no_delete
    BEFORE DELETE OR TRUNCATE ON current_index
    FOR EACH STATEMENT EXECUTE FUNCTION forbid_mutation();
