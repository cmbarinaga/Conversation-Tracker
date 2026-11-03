#!/usr/bin/env bash
# Run from any directory in a Linux Codex cloud environment.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo 'uv is required. Install uv, then rerun: bash scripts/setup-cloud.sh' >&2
  exit 1
fi
if ! command -v node >/dev/null 2>&1; then
  echo 'Node 22+ is required for the browser calculation tests.' >&2
  exit 1
fi
node -e 'if (Number(process.versions.node.split(".")[0]) < 22) { console.error("Node 22+ is required."); process.exit(1); }'

uv sync --frozen --python 3.12
uv run --frozen --python 3.12 python -m unittest discover -s tests -v
node --test tests/thread_math.test.js
uv run --frozen --python 3.12 python conversation_tracker.py --no-open

echo 'Ready: output/conversation_tracker.html'
