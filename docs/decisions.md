# Decisions

One entry per decision, with the reason. Entries are never edited after the fact; a change of mind is a new entry that names the one it replaces. A run on the test set must cite a heading from this file (`make confirm-test DECISION=D-0xx`).

Status tags: **default** means the guide's recommended default, not yet confirmed; **open** means only you can answer it. Replace the tag with **confirmed** and a date when you answer.

## W0-1 answers

### D-001 Corpus

PostgreSQL 16 manual subset, chosen by chapter in `corpus/chapters.txt` (SQL language, server administration, index internals). Permissive PostgreSQL License, stable section ids, answers I can verify. **default**: check the licence page before the first commit. The page list is generated, not typed: `rag corpus discover` reads each chapter's table of contents, so no URL is guessed.

### D-002 Generation model

Hosted model (`RAG_GENERATOR=anthropic`) with the model id pinned in `RAG_GEN_MODEL` and a hard cap in `RAG_GEN_TOKEN_BUDGET`; answers cached by input hash so reruns cost nothing. Retrieval is evaluated with no model at all. **default**: set the model id and the cap.

### D-003 Time

Start and finish weeks, hours per week. **open**

### D-004 Target roles

Data engineering, machine learning engineering, or both. Changes which results lead the README, not the work. **open**

### D-005 Machine

Computer that runs the evaluation (model, CPU, memory) for the README. **open**

### D-006 Repository and licence

`rag-eval-pipeline`, MIT, public, under my personal GitHub account. **default**: put your name in `LICENSE` before the first commit.

### D-007 Employer policy

Personal projects allowed; built on a personal device and account, no work credentials. **open**: must be answered before the first commit.

## Design decisions

### D-008 Gold labels are the manual's own section anchors

A gold section is `page#anchor`, for example `indexes-types#indexes-types-btree`, taken from the `id` of the manual's section `div`, lowercased. Every chunk stores the refs of its section and all its ancestors, so a gold label on a top-level section matches chunks from its subsections. Anchors are stable across manual rebuilds; numbered titles are not.

### D-009 Tokens are counted with a regex, not the model tokenizer

Words and punctuation marks count as one token each. Chunking stays deterministic and independent of the embedding model, so changing the model does not change chunk ids. bge-small truncates at 512 word pieces, so the 800-token variant is partly truncated; the ablation shows that cost instead of hiding it.

### D-010 Chunk ids include the chunker key

`chunk_id = sha256(chunker key, document sha256, section ref, start offset)`. The guide's three inputs plus the chunker version and parameter hash, so chunk sets from different chunkers live in one table without collisions.

### D-011 One partition and one HNSW index per index version

Embeddings live in a table partitioned by `index_id`. A build loads a standalone table, builds its HNSW index, inserts the `index_versions` row and attaches the partition in one transaction, so a crashed build leaves nothing behind and a rebuild of the same id is a no-op. Queries read the partition directly so the planner uses its HNSW index. The schema fixes vectors at 384 dimensions; a different model needs a migration.

### D-012 Sections and chunks are ingested separately

`ingest` parses each verified document into sections once per parser version; chunks are made per chunker config. Each step writes a receipt in the same transaction as its rows, so a rerun skips finished work and a killed run resumes cleanly.

### D-013 nDCG is section-level

Each gold section earns gain once, at the first rank that covers it; the ideal is `min(gold sections, 10)` hits at the top. Chunk-level relevance would reward returning five chunks of the same section.

### D-014 Unanswerable questions are scored only on abstention

They have no gold sections, so they are left out of retrieval metrics and scored by abstention accuracy when answers are generated.

### D-015 The gate fails closed and floors only go up

No floors in `eval/gate.toml` means the gate refuses to run. Floors are checked against the last recorded decision and by `rag gate set-floors`; lowering one is an error. Regression tolerance is 0.02 (2 points on a 0 to 1 scale) against the served index on the same question set; a served index judged on an older question set must be re-evaluated first.

### D-016 The database, not the code, makes publication safe

`publications` has a composite foreign key to `(decision_id, index_id, retriever_key, passed)` in `gate_decisions` plus `CHECK (passed)`, so only a passing decision can be published. A trigger re-checks this on the current-index pointer, and history tables are append-only by trigger. A decision records which publication was serving when it was made; if that changed, publishing it is refused.

### D-017 The retriever is part of the run identity, not the index identity

An index id covers corpus, chunker, embedding model and revision, and HNSW parameters. The retriever config (mode, candidate depth, fusion constant, re-ranker model and revision) goes into the run key, the gate decision and the publication. One index can be evaluated and served with different retrievers without rebuilding it, and results from different configs are never mixed.

### D-018 The test split is locked twice

At runtime, only `rag.evaluate.confirm` can create the access token that `load_test` requires. Statically, a unit test fails if any other module refers to it. Each test-set run must cite a new heading in this file and is appended to `evidence/test_confirmation.json`.

### D-019 A hashing embedder for tests and CI

A deterministic bag-of-words hashing embedder with the same 384 dimensions, so tests and the CI evaluation run with no network and no model download. Its scores are never reported as results.

### D-020 Re-ranker revision must be pinned before use

`hybrid-rerank` uses `cross-encoder/ms-marco-MiniLM-L-6-v2`; its `rerank_revision` in `experiments.toml` is empty, and the code refuses to run an unpinned re-ranker. Pin the commit sha from the model page before R17.

### D-021 Answer metrics use the top 5 retrieved chunks

The generator sees the top 5 chunks. Citation validity is per answer: an answerable question answered without citations scores 0, and a citation counts only if the chunk was in that context and covers a gold section.

### D-022 Generation lives in its own package

`src/rag/generate/` is an addition to the guide's layout. It keeps the model behind one interface used by both the evaluation runner and the API: an extractive baseline with no model, and the hosted adapter with cache and spending cap.

### D-023 Pinned images and models

`pgvector/pgvector:0.8.0-pg16` in Compose and CI; bge-small-en-v1.5 at revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. On Linux the `models` extra installs PyTorch from PyPI, which is large; acceptable for one laptop.

### D-024 The page list is generated from chapters

`corpus/chapters.txt` names the chapters in scope; `rag corpus discover` expands each into its pages and writes `corpus/sources.txt`. A first hand-typed list of 36 pages gave 167 chunks, too few for the comparisons in the ablation to mean much, and a hand-typed list risks wrong URLs. The manifest stays the frozen record: it is written once by `corpus init` from the generated list. Scope is changed by editing the chapter list and making a new manifest before the first commit, never by editing a manifest.

### D-025 The question set was drafted from the ingested sections and is checked by script
100 questions in `eval/questions.jsonl`: 50 factual, 20 paraphrase (written in everyday wording, avoiding the manual's terms), 20 multi_section (two gold sections each) and 10 unanswerable (topics the corpus does not cover, such as PgBouncer or Patroni, confirmed absent by text search). Drafted by the AI assistant from an export of the ingested sections, then checked by script: every gold section exists and every key fact is a verbatim phrase of its gold sections' text. Known biases: the author saw the sections, so factual questions often share words with their answer and are easier than real user queries (the paraphrase type exists to show that gap); three factual questions list a second gold section where two sections state the same fact; key facts written as words ("four megabytes") will not match an answer that writes "4 MB". **open**: the gold labels count as my own only after I review them with the review sheet, before `rag eval split` freezes the split.

### D-026 One gold label revised after the first dev run
`pg-p-004` ("What guarantees my data survives if the server loses power right after I commit?") listed only `wal-intro#wal-intro`. After the first ablation every chunker missed it at recall@10, and the retrieved `wal-reliability#wal-reliability` states in its first paragraph that committed data must be stored "safe from power loss". Both sections answer the question, so the gold now lists both and the key fact is "permanent storage", which both state. Judged on the question's merit, not on how it scored; the other misses (`pg-f-026`, `pg-f-006`, `pg-p-007`) are retrieval failures with a single correct section and keep their labels. Done before the gate floors were set and before the test split was opened; the split is recomputed with the same seed and the same ids, so the dev/test assignment does not change.

### D-027 Gate floors come from the default preset, with a margin of one question
The baseline is heading-400 with the vector retriever (eval 28), the default preset in experiments.toml, fixed before the ablation, not the best of the 12 configurations: picking the maximum on 27 dev questions would build selection luck into floors that can never be lowered. Floors are the baseline minus 0.05, a little more than the 0.037 that one dev question moves recall, so losing one question does not fail the gate and losing two does. The regression tolerance of 0.02 against the served index stays for now, but it is smaller than one question for the recall metrics; a new entry superseding D-015 should settle it before the gate is relied on.

### D-028 Re-ranker pinned to a commit
The hybrid-rerank retriever uses cross-encoder/ms-marco-MiniLM-L-6-v2 at revision 233902d25c440f23af6f7d6e94d2946bac0bee0a, read from the Hugging Face model page on 2026-10-15 and written into experiments.toml (resolves the open point in D-020). It is evaluated on all four chunkers like the other retrievers. If it earns a place it goes through the same gate as any candidate.

### D-029 Keep heading-800 with vector serving; one test-set run on it
Dev ablation with the re-ranker (D-028): hybrid-rerank on heading-400 scores best (recall@5 1.000, nDCG@10 0.845) against 0.926 and 0.822 for the served heading-800 with vector, at roughly 15 to 20 times the retrieval latency (p95 about 490 ms against 26 ms). On 27 answerable dev questions a gap of 0.023 nDCG and two questions at recall@5 is within what one or two questions can produce, so the simpler and faster configuration stays and no re-ranker is served. The re-ranker is better on heading-400 and fixed-400 and slightly worse on heading-800 and heading-200; it stays in the ablation as a measured trade-off. The regression tolerance stays at 0.02: it is stricter than one question for the recall metrics, so the gate fails closed, which is intended (a false rejection costs a rerun, a false acceptance costs quality). The test set is opened once, for the served configuration (index 57235aed6e4d42dd, vector retriever), and the result is reported whatever it is. No label, configuration or floor changes after it.

### D-030 Time (confirms D-003)

Started 8 September 2026, finished 14 October 2026: about five and a half weeks. Hours per week were not tracked and are not claimed. Confirmed 15 October 2026.

### D-031 Target roles (confirms D-004)

Both data engineering and machine learning engineering. The README leads with the evaluation gate and the database-enforced publication (the data-engineering side) and with the retrieval ablation and evaluation design (the ML side). Confirmed 15 October 2026.

### D-032 Machine (confirms D-005)

MacBook Pro, Apple M1 Pro, 16 GB, CPU inference, PostgreSQL 16 in Docker Desktop. All latency and build times in the README come from this machine. Confirmed 15 October 2026.

### D-033 Answer quality is reported as an extractive baseline

The released answer metrics (dev set) come from the extractive-v1 generator: key-fact coverage 0.22, citation validity 0.37, false abstention 0.41, abstention accuracy 0.67. Reading all 30 dev answers shows two failures: the right section is retrieved but the generator selects neighbouring sentences (9 of 16 answered questions), and it abstains on 11 of 27 answerable questions, mostly paraphrases and multi-section questions whose wording differs from the manual. Retrieval and generation are reported separately, the gate judges retrieval only, and no hosted-model run is part of this release. D-002 stays the default configuration and is not exercised here. The test set was spent on retrieval (D-029) and is not used for answers. Confirmed 15 October 2026.

### D-034 Repository and licence (confirms D-006)

`rag-eval-pipeline`, MIT licence, copyright Nikita Chevokin, public. Confirmed 15 October 2026.
