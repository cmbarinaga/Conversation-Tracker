# Conversation Tracker

## Current state

Steps 1, 2, and 3 are implemented. `Handover.md`, `Step2.md`, and `Step3.md` are historical requirements, not pending tasks. Read `README.md` and `CLOUD_HANDOFF.md` before changing behavior. User instructions take precedence over historical scope limits.

The default viewer is future-aware: 128 past / 48 future model tokens, passage embeddings, separate alignment scores, and delayed playback. Original causal (K=5) and PCA views remain available. Keep every transcript sentence, including acknowledgments. Do not introduce filler filtering, LLM calls, Android, or speech recognition unless requested.

## Run and verify

- Python 3.12, Node 22+, and uv. Dependencies are pinned in `uv.lock` (CPU PyTorch).
- Install: `uv sync --frozen --python 3.12`.
- Python tests: `uv run --frozen --python 3.12 python -m unittest discover -s tests -v`.
- Browser math/playback tests: `node --test tests/thread_math.test.js`.
- Full pipeline: `uv run --frozen --python 3.12 python conversation_tracker.py --no-open`.
- Linux cloud setup: `bash scripts/setup-cloud.sh` (install, tests, generation).
- Temporary preview: `uv run --python 3.12 python -m http.server 8765 --bind 0.0.0.0 --directory output`; use the cloud environment's preview/port access and stop the server after testing.

Model/data downloads need internet initially. The resulting HTML is self-contained. Do not silently replace real embeddings or conversations when downloads fail. Cloud tasks have no local desktop browser; use `--no-open`.

## Structure and constraints

- `conversation_tracker.py`: AMI parsing, individual sentence embeddings, PCA, offline HTML bundling.
- `semantic_context.py`: actual MiniLM tokenizer, bounded past/future passages, passage embedding and audit exports. Token budgets include special tokens; guard against silent truncation.
- `web/thread_math.js`, `web/playback.js`: browser/Node math and deterministic playback. `web/viewer.js` and `web/viewer.html`: GUI, inlined at generation.
- `output/`, `.cache/`, and `.venv/` are local/generated and ignored.
- `sample-output/` is a committed baseline snapshot of completed work, including HTML, arrays, audits, and previews. Normal tests/generation should not rewrite it. Refresh it only when explicitly documenting a new baseline.

Preserve array/record ordering and the original causal comparison. Past/future scores must not be averaged. Completed geometry must remain fixed during playback, including after buffering releases. Replay and scrubbing must be deterministic.

The turn rule is experimental. Reduced angles do not establish improved topic-change accuracy: the genuine slogan transition also softened. Keep this limitation visible in documentation and reports. No extensive evaluation infrastructure is needed for routine changes.

Project code is MIT licensed. AMI sample material is CC BY 4.0; retain `AMI-LICENCE.txt` and `SOURCE.md` with any shared sample data. The model is Apache 2.0, and model weights are downloaded separately.
