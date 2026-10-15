import argparse
import getpass
import json
import sys
from collections.abc import Callable

import httpx

from rag.config import Paths, database_url, default_paths
from rag.db import connect, migrate


def _conn():
    return connect(database_url())


def _experiments(paths: Paths):
    from rag.specs import load_experiments

    return load_experiments(paths.experiments)


def _manifest(paths: Paths):
    from rag.ingest.manifest import load_manifest

    if not paths.manifest.exists():
        sys.exit(f"{paths.manifest} not found; create it once with `rag corpus init`")
    return load_manifest(paths.manifest)


def _index_id_for(paths: Paths, preset: str | None) -> tuple[str, object]:
    from rag.index.identity import index_id, index_identity

    spec = _experiments(paths).index(preset)
    return index_id(index_identity(_manifest(paths).corpus_sha256, spec)), spec


def cmd_db_migrate(args, paths: Paths) -> None:
    with _conn() as conn:
        applied = migrate(conn, paths.migrations)
    print(f"applied migrations: {applied or 'none (up to date)'}")


def cmd_corpus_discover(args, paths: Paths) -> None:
    from rag.ingest.discover import discover, write_sources
    from rag.ingest.manifest import http_fetcher, read_sources

    chapters = read_sources(paths.chapters)
    urls = discover(chapters, http_fetcher())
    write_sources(paths.sources, urls, chapters)
    print(
        f"{len(urls)} pages from {len(chapters)} chapters written to "
        f"{paths.sources.relative_to(paths.root)}"
    )
    if paths.manifest.exists():
        print(
            f"note: {paths.manifest.relative_to(paths.root)} already exists and was not touched; "
            "to use the new list, delete it and data/raw, then run `make corpus-init`"
        )


def cmd_corpus_init(args, paths: Paths) -> None:
    from rag.ingest.manifest import build_manifest, http_fetcher, read_sources

    manifest = build_manifest(
        read_sources(paths.sources),
        corpus=args.corpus,
        license=args.license,
        source=args.source,
        raw_dir=paths.raw_dir,
        out=paths.manifest,
        fetcher=http_fetcher(),
    )
    print(f"wrote {paths.manifest}: {len(manifest.files)} files, corpus {manifest.corpus_sha256}")


def cmd_corpus_fetch(args, paths: Paths) -> None:
    from rag.ingest.manifest import fetch_corpus, http_fetcher, verify_corpus

    manifest = _manifest(paths)
    downloaded = fetch_corpus(manifest, paths.raw_dir, http_fetcher())
    verify_corpus(manifest, paths.raw_dir)
    print(f"downloaded {len(downloaded)} files; all {len(manifest.files)} verified")


def cmd_corpus_verify(args, paths: Paths) -> None:
    from rag.ingest.manifest import verify_corpus

    manifest = _manifest(paths)
    verify_corpus(manifest, paths.raw_dir)
    print(f"verified {len(manifest.files)} files against corpus {manifest.corpus_sha256}")


def cmd_ingest(args, paths: Paths) -> None:
    from rag.ingest.load import ingest

    experiments = _experiments(paths)
    presets = args.index or [experiments.default_index]
    chunkers = {experiments.index(p).chunker.key: experiments.index(p).chunker for p in presets}
    with _conn() as conn:
        report = ingest(conn, _manifest(paths), paths.raw_dir, chunkers.values())
    print(
        json.dumps(
            {
                "corpus_sha256": report.corpus_sha256,
                "documents_added": report.documents_added,
                "sections_added": report.sections_added,
                "chunks_added": report.chunks_added,
            },
            indent=2,
        )
    )


def cmd_index_build(args, paths: Paths) -> None:
    from rag.index.build import build_index
    from rag.index.embed import make_embedder

    spec = _experiments(paths).index(args.index)
    with _conn() as conn:
        result = build_index(
            conn, _manifest(paths).corpus_sha256, spec, make_embedder(spec.embedder)
        )
    state = "built" if result.created else "already exists (no-op)"
    print(
        f"index {result.index_id} [{spec.name}] {state}: {result.n_chunks} chunks, "
        f"{result.build_seconds}s"
    )


def cmd_index_list(args, paths: Paths) -> None:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT index_id, chunker_key, embed_model, n_chunks, build_seconds, built_at::text "
            "FROM index_versions ORDER BY built_at"
        ).fetchall()
    for r in rows:
        print(
            f"{r['index_id']}  {r['chunker_key']:<28} {r['embed_model']:<28} "
            f"{r['n_chunks']:>6} chunks  {r['build_seconds']:>8.1f}s  {r['built_at']}"
        )


def cmd_eval_validate(args, paths: Paths) -> None:
    from rag.evaluate.questions import load_question_file, type_mix, validate_questions
    from rag.ingest.load import corpus_section_refs

    questions = load_question_file(paths.questions)
    with _conn() as conn:
        refs = corpus_section_refs(conn, _manifest(paths).corpus_sha256)
    if not refs:
        sys.exit("no sections in the database for this corpus; run `make ingest` first")
    problems = validate_questions(questions, refs)
    print(f"{len(questions)} questions, type mix {type_mix(questions)}")
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("all gold sections exist and every answerable question has key facts")


def cmd_eval_split(args, paths: Paths) -> None:
    from rag.evaluate.questions import load_question_file, make_split, type_mix, write_split

    questions = load_question_file(paths.questions)
    existing = json.loads(paths.split.read_text()) if args.extend and paths.split.exists() else None
    seed = existing["seed"] if existing else args.seed
    split = make_split(questions, seed=seed, existing=existing)
    write_split(paths.split, split, extend=args.extend)
    by_id = {q.id: q for q in questions}
    print(f"dev {len(split['dev'])} {type_mix([by_id[i] for i in split['dev']])}")
    print(f"test {len(split['test'])} {type_mix([by_id[i] for i in split['test']])}")


def cmd_eval_run(args, paths: Paths) -> None:
    from rag.evaluate.questions import load_dev
    from rag.evaluate.runner import run_evaluation
    from rag.generate.base import make_generator
    from rag.index.embed import make_embedder

    iid, spec = _index_id_for(paths, args.index)
    retriever = _experiments(paths).retriever(args.retriever)
    questions = load_dev(paths.questions, paths.split)
    with _conn() as conn:
        result = run_evaluation(
            conn,
            index_id=iid,
            retriever_spec=retriever,
            questions=questions,
            split="dev",
            embedder=make_embedder(spec.embedder),
            evidence_dir=paths.evidence,
            generator=make_generator(conn=conn) if args.answers else None,
            labels={"index": spec.name, "retriever": retriever.name},
        )
    summary = {k: v for k, v in result.metrics.items() if k != "by_type"}
    print(f"eval {result.eval_id} run {result.run_key}: {json.dumps(summary)} {result.timing}")
    print(f"evidence: {result.evidence_path.relative_to(paths.root)}")


def cmd_eval_confirm_test(args, paths: Paths) -> None:
    from rag.evaluate.confirm import confirm_on_test
    from rag.index.embed import make_embedder

    iid, spec = _index_id_for(paths, args.index)
    retriever = _experiments(paths).retriever(args.retriever)
    with _conn() as conn:
        result = confirm_on_test(
            conn,
            decision=args.decision,
            questions_path=paths.questions,
            split_path=paths.split,
            decisions_md=paths.decisions,
            evidence_dir=paths.evidence,
            index_id=iid,
            retriever_spec=retriever,
            embedder=make_embedder(spec.embedder),
            labels={"index": spec.name, "retriever": retriever.name},
        )
    print(f"test-set run recorded under {args.decision}: {json.dumps(result.metrics)}")


def cmd_gate_set_floors(args, paths: Paths) -> None:
    from rag.evaluate.gate import (
        GateConfig,
        floors_from_baseline,
        load_gate_config,
        render_gate_config,
    )

    config = load_gate_config(paths.gate)
    with _conn() as conn:
        row = conn.execute(
            "SELECT metrics, split FROM evaluations WHERE eval_id = %s", (args.eval_id,)
        ).fetchone()
    if row is None or row["split"] != "dev":
        sys.exit(f"evaluation {args.eval_id} is not a dev-set evaluation")
    floors = floors_from_baseline(row["metrics"], args.margin)
    lowered = [m for m, v in floors.items() if m in config.floors and v < config.floors[m]]
    if lowered:
        sys.exit(f"refusing to loosen floors for {lowered}; floors only ever go up")
    paths.gate.write_text(
        render_gate_config(GateConfig(floors, config.regression_tolerance), args.eval_id)
    )
    print(
        f"wrote floors {floors} to {paths.gate.relative_to(paths.root)}; commit it with the "
        "baseline evidence file"
    )


def cmd_gate_check(args, paths: Paths) -> None:
    from rag.evaluate.gate import check_gate, load_gate_config

    config = load_gate_config(paths.gate)
    with _conn() as conn:
        eval_id = args.eval_id
        if args.latest:
            row = conn.execute(
                "SELECT max(eval_id) AS id FROM evaluations WHERE split = 'dev'"
            ).fetchone()
            eval_id = row["id"]
        if eval_id is None:
            sys.exit("no dev evaluation to judge")
        decision = check_gate(conn, eval_id, config)
    verdict = "PASS" if decision.passed else "FAIL"
    print(f"gate decision {decision.decision_id} for eval {eval_id}: {verdict}")
    for reason in decision.reasons:
        print(f"  - {reason}")
    if not decision.passed:
        sys.exit(2)


def cmd_publish(args, paths: Paths) -> None:
    from rag.publish.switch import latest_passing_decision, publish

    with _conn() as conn:
        decision_id = args.decision or latest_passing_decision(conn)
        if decision_id is None:
            sys.exit("no unpublished passing gate decision")
        publication_id = publish(conn, decision_id, args.by)
    print(f"publication {publication_id}: decision {decision_id} is now serving")


def cmd_status(args, paths: Paths) -> None:
    from rag.publish.switch import current

    with _conn() as conn:
        served = current(conn)
    if served is None:
        print("nothing published yet")
        return
    print(
        f"serving index {served.index_id} with retriever {served.retriever.get('name')} "
        f"(publication {served.publication_id}, decision {served.decision_id}, "
        f"since {served.published_at})"
    )


def cmd_report_ablation(args, paths: Paths) -> None:
    from rag.evaluate.report import ablation_markdown

    out = paths.evidence / "ablation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ablation_markdown(paths.evidence))
    print(f"wrote {out.relative_to(paths.root)}")


def cmd_serve(args, paths: Paths) -> None:
    import uvicorn

    uvicorn.run("rag.api.app:create_app", factory=True, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="group", required=True)

    def add(group, name: str, fn: Callable, help_: str):
        p = group.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
        return p

    db = sub.add_parser("db").add_subparsers(dest="cmd", required=True)
    add(db, "migrate", cmd_db_migrate, "apply forward-only migrations")

    corpus = sub.add_parser("corpus").add_subparsers(dest="cmd", required=True)
    add(
        corpus,
        "discover",
        cmd_corpus_discover,
        "expand corpus/chapters.txt into corpus/sources.txt",
    )
    p = add(corpus, "init", cmd_corpus_init, "download sources.txt once and write the manifest")
    p.add_argument("--corpus", default="postgresql-16-manual-subset")
    p.add_argument("--license", default="PostgreSQL License")
    p.add_argument("--source", default="https://www.postgresql.org/docs/16/")
    add(corpus, "fetch", cmd_corpus_fetch, "download missing files by manifest and verify")
    add(corpus, "verify", cmd_corpus_verify, "check every file against its sha256")

    p = add(sub, "ingest", cmd_ingest, "parse and chunk the verified corpus")
    p.add_argument("--index", action="append", help="index preset whose chunker to run")

    index = sub.add_parser("index").add_subparsers(dest="cmd", required=True)
    p = add(index, "build", cmd_index_build, "embed chunks and build an index version")
    p.add_argument("--index")
    add(index, "list", cmd_index_list, "list index versions")

    ev = sub.add_parser("eval").add_subparsers(dest="cmd", required=True)
    add(ev, "validate", cmd_eval_validate, "check questions against the ingested corpus")
    p = add(ev, "split", cmd_eval_split, "freeze the dev/test split")
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--extend", action="store_true", help="assign newly added questions only")
    p = add(ev, "run", cmd_eval_run, "score an index on the dev split")
    p.add_argument("--index")
    p.add_argument("--retriever")
    p.add_argument("--answers", action="store_true", help="also generate and score answers")
    p = add(ev, "confirm-test", cmd_eval_confirm_test, "run the chosen config on the test set")
    p.add_argument("--index", required=True)
    p.add_argument("--retriever", required=True)
    p.add_argument("--decision", required=True, help="decision entry id, e.g. D-012")

    gate = sub.add_parser("gate").add_subparsers(dest="cmd", required=True)
    p = add(gate, "set-floors", cmd_gate_set_floors, "write floors from a baseline evaluation")
    p.add_argument("--eval-id", type=int, required=True)
    p.add_argument("--margin", type=float, default=0.03)
    p = add(gate, "check", cmd_gate_check, "judge an evaluation and record the decision")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--eval-id", type=int)
    group.add_argument("--latest", action="store_true")

    p = add(sub, "publish", cmd_publish, "atomically switch the served index")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--decision", type=int)
    group.add_argument("--latest-pass", action="store_true")
    p.add_argument("--by", default=getpass.getuser())

    add(sub, "status", cmd_status, "show the served index")

    report = sub.add_parser("report").add_subparsers(dest="cmd", required=True)
    add(report, "ablation", cmd_report_ablation, "generate evidence/ablation.md")

    p = add(sub, "serve", cmd_serve, "run the API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.fn(args, default_paths())
    except (
        RuntimeError,
        ValueError,
        PermissionError,
        LookupError,
        FileExistsError,
        httpx.HTTPError,
    ) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
