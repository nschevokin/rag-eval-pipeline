# rag-eval-pipeline
![ci](https://github.com/nschevokin/rag-eval-pipeline/actions/workflows/ci.yml/badge.svg)
Question answering over a fixed slice of the PostgreSQL 16 manual, where a new search index serves users only after it passes a measured evaluation gate. Retrieval quality is measured, versioned and enforced by the database, not assumed.

> Status: complete first version. Every number in this README points to a file in `evidence/`.

## How it works

```mermaid
flowchart LR
    M[Corpus manifest<br/>sha256 per file] --> I[Ingest and chunk]
    I --> B[Embed and build index<br/>id = hash of inputs]
    B --> E[Evaluate on dev questions]
    E --> G{Gate passes?}
    G -- no --> R[Rejected; old index serves]
    G -- yes --> P[Publish: atomic switch]
    P --> A[Answers with citations]
```

All state lives in PostgreSQL 16 with pgvector: documents, sections, chunks, index versions, evaluations, gate decisions, publications and the current-index pointer.

- **Immutable inputs.** `corpus/manifest.json` lists every file with its sha256 and is never edited. `rag corpus verify` names any file that changed.
- **Index identity.** An index id is a hash of the corpus hash, chunker version and parameters, embedding model and revision, and HNSW parameters. Same inputs, same id (a rebuild is a no-op); any change, new id. Each index version is its own partition with its own HNSW index.
- **Evaluation.** A hand-written question set with gold sections (the manual's own section anchors) and key facts, split once with a fixed seed into dev (30%) and test (70%). Experiments read only dev. The test split can be opened only by `rag eval confirm-test`, once per recorded decision.
- **Gate.** Absolute floors in `eval/gate.toml`, set from the first baseline and never loosened, plus a regression rule: no retrieval metric more than 2 points below the served index. Decisions are written to the database with their reasons.
- **Publication.** One transaction moves the current-index pointer. Foreign keys and a trigger make it impossible to publish an index without a passing gate decision, even with hand-written SQL.

## Quick start

Needs Docker, [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
make setup                   # pinned dependencies, including the embedding model
make db                      # PostgreSQL 16 + pgvector on localhost:5433
make discover                # expand corpus/chapters.txt into the page list
make corpus-init             # once only: download those pages, write and commit the manifest
make ingest                  # migrate, download by manifest, verify, parse, chunk
make build-index             # INDEX=heading-400 by default
make evaluate                # dev-set scores, then the gate
make publish                 # serve the latest passing decision
make serve                   # API on http://127.0.0.1:8000
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "When does a partial index help?"}'
```

The answer carries its citations (section anchor and source URL) and the id of the index that served it.

## Commands

| Command | What it does |
|---|---|
| `rag corpus discover` | Expand `corpus/chapters.txt` into `corpus/sources.txt` using each chapter's table of contents |
| `rag corpus init` | One-off: download `corpus/sources.txt`, write the manifest (refuses to overwrite) |
| `rag corpus fetch` / `verify` | Download missing files by manifest / check every sha256 |
| `rag ingest --index P` | Parse and chunk with the chunker of preset P; idempotent |
| `rag index build --index P` | Embed and build; no-op if the id exists |
| `rag eval validate` | Every gold section exists; every answerable question has key facts |
| `rag eval split` | Freeze the dev/test split (`--extend` for newly added questions) |
| `rag eval run --index P --retriever R [--answers]` | Score on dev; write the evaluation row and `evidence/evals/*.json` |
| `rag gate set-floors --eval-id N` | Floors = baseline minus a margin; refuses to lower any floor |
| `rag gate check --latest` | Record pass or fail with reasons |
| `rag publish --latest-pass` | Atomic switch of the served index |
| `rag report ablation` | Generate `evidence/ablation.md` from the evidence files |
| `rag eval confirm-test --decision D-0xx` | The single test-set run for a recorded decision |

Presets for chunkers, embedders, indexes and retrievers are in `experiments.toml`.

## Results

Served configuration: the `heading-800` index (638 chunks from 203 pages) with the `vector` retriever. It was chosen on the dev set by the reasoning in [D-027 to D-029](docs/decisions.md); the test set was opened once, for this configuration only (D-029), and the numbers are reported as they came out.

| | Dev (27 questions) | Test (63 questions) |
|---|---|---|
| Recall@5 | 0.926 | 0.952 |
| Recall@10 | 0.963 | 0.984 |
| MRR | 0.811 | 0.843 |
| nDCG@10 | 0.822 | 0.842 |

Sources: `evidence/evals/dev-57235aed6e4d42dd-c6d2ea78ce50.json`, `evidence/evals/test-57235aed6e4d42dd-c6d2ea78ce50.json`, `evidence/test_confirmation.json`. Questions that have no answer in the manual are not part of these counts (D-014). One question moves recall by 3.7 points on dev and 1.6 on test; with 60 of 63 test questions found in the top 5, the 95% interval for test recall@5 is about 0.87 to 0.98. Test scores came out slightly above dev scores, so there is no sign that choosing a configuration on dev overfitted to it, but the gap is within noise.

By question type on the test set (from `evidence/evals/test-57235aed6e4d42dd-c6d2ea78ce50.json`; 14 to 35 questions per type, so read these as indications only):

| Type | n | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| factual | 35 | 0.971 | 0.971 | 0.912 | 0.922 |
| multi_section | 14 | 0.929 | 1.000 | 0.818 | 0.714 |
| paraphrase | 14 | 0.929 | 1.000 | 0.695 | 0.771 |

Paraphrased questions (everyday wording) find the right section but rank it lower (MRR 0.70 against 0.91 for factual). Recall counts a hit if any gold section is retrieved, so multi-section questions look perfect at recall@10 while nDCG shows that finding all the gold sections is harder.

### Ablation (dev set)

16 combinations of 4 chunkers and 4 retrievers are in [`evidence/ablation.md`](evidence/ablation.md), generated from the evidence files. What it shows:

- Full-text search alone (Postgres `ts_rank_cd` over an OR query) is the weakest retriever for every chunker, nDCG@10 0.56 to 0.68 against 0.77 to 0.82 for vector.
- Hybrid search (reciprocal rank fusion of vector and full-text) did not beat vector alone.
- The vector results across chunk sizes are within a question or two of each other on 27 questions, so no chunk size is claimed to be better.
- The cross-encoder re-ranker gave the best dev scores on `heading-400` (recall@5 1.000, nDCG@10 0.845) and `fixed-400`, slightly worse ones on `heading-800` and `heading-200`, and made retrieval roughly 19 times slower. It was not adopted (D-029).

### Latency and build time

Measured on one machine: MacBook Pro, Apple M1 Pro, 16 GB, CPU inference, local PostgreSQL 16 in Docker. Retrieval p95 for the served configuration is 22 to 26 ms including embedding the query; with the re-ranker it is 400 to 500 ms. Building the served index took 14.2 s (638 chunks); the `heading-400` index (1,008 chunks) took 37.7 s.

### Answer quality (dev set, extractive baseline)

With the no-model extractive generator (picks the sentences that overlap most with the question and abstains below a fixed overlap): key-fact coverage 0.22, citation validity 0.37, false abstention 0.41, abstention accuracy 0.67 (2 of 3 unanswerable questions correctly declined). Reading all 30 dev answers (`evidence/evals/dev-57235aed6e4d42dd-c6d2ea78ce50-answers.json`) shows two separate failures, and neither is a retrieval failure:

- **Wrong sentences from the right section.** Of the 16 answered questions, 5 contained every key fact and 2 contained half; the other 9 quoted neighbouring sentences instead (for the shared_buffers question the right section is ranked first, but the answer gives the 25% rule of thumb, not the default value).
- **Too many abstentions.** 11 of 27 answerable questions were declined: 5 of 6 paraphrases, 4 of 6 multi-section questions and 2 of 15 factual ones. Questions that share few words with the manual fall below the fixed overlap threshold. The one unanswerable question that was answered anyway (pg-u-009) shares words with the manual's operator-class tables.

The hosted-model adapter exists (`RAG_GENERATOR=anthropic`, with cache and spending cap) but was not evaluated in this release (D-033). The test set was spent on retrieval, so these are dev-set numbers.

### The gate in action

A full-text candidate on the `heading-800` index was judged by the gate and rejected with eight reasons (every metric below its floor and more than 0.02 below the served index). `rag publish` refused it, and a hand-written `INSERT INTO publications` was refused by the database itself (`violates foreign key constraint ... Key (decision_id, index_id, retriever_key, passed)=(3, ..., t) is not present in table "gate_decisions"`). The passing candidate was then published with one atomic switch.

## Tests

```bash
make test-unit                                                   # no database needed
RAG_TEST_DATABASE_URL=postgresql://rag:rag@localhost:5433/postgres make test
```

Unit tests run with no network and no model download. Integration tests create a throwaway database per test from a migrated template and use a deterministic hashing embedder on a tiny synthetic corpus in `tests/fixtures/`. Each claim above has a test: tampered corpus rejected, deterministic chunk ids, idempotent and crash-safe ingestion, index identity, hand-computed metrics and fusion order, no dev/test leakage, a deliberately broken index rejected by the gate, publication without a passing gate refused by the database, a crash before commit leaving the old index serving, and answer metrics that cannot be fooled by invented citations.

## Limits

- One small corpus: 203 pages of the PostgreSQL 16 manual (454 sections), on one machine. Nothing here is a claim about scale or production use.
- The question set is small (100 questions; 27 answerable on dev, 63 on test). Differences of a few points between configurations are noise, and the gate's regression tolerance (0.02) is smaller than one dev question's worth of recall (0.037), so it fails closed: a candidate that loses a single question is rejected.
- The questions were drafted with an AI assistant from the ingested text. Every key fact is checked by script to appear verbatim in its gold section ([D-025](docs/decisions.md)), and one gold label was revised after the first dev run ([D-026](docs/decisions.md)). Factual questions often share words with the manual, so they overstate performance on real user questions; the paraphrase type is the harder indicator.
- The gate only sees the questions in the set. It does not catch questions outside it, drift in the corpus itself, or answer problems that key-fact matching cannot see.
- Key-fact coverage is string and number matching: it can miss a correct paraphrase (a fact written "four megabytes" will not match "4 MB") and credit an answer that states a fact in a misleading context.
- The parser keeps the manual's `#` permalink character in heading text, which ends up in the embedded text. Fixing it would change every chunk and invalidate the baseline, so it is recorded as a known flaw instead.
- The answer generator is a deliberately simple extractive baseline; its answer metrics are low and its two failure modes (wrong sentences, over-abstaining on paraphrases) are described above.
- HNSW is approximate and its graph build is randomised, so a rebuild on another machine can differ slightly; reproduced scores are compared within a stated tolerance.

## Corpus licence

The corpus is a subset of the PostgreSQL 16 documentation, © The PostgreSQL Global Development Group, distributed under the PostgreSQL License. This repository commits the manifest (file names, URLs and hashes), not the manual itself; the pages are downloaded by manifest. The answer evidence file contains short excerpts of the manual (the extractive answers), reproduced under the same licence and notice.

## How it was built

Built with an AI assistant as implementation partner. Every design choice is recorded with its reason in [`docs/decisions.md`](docs/decisions.md).

## Licence

MIT, see [`LICENSE`](LICENSE).
