# Continue in Codex cloud

Repository: [cmbarinaga/Conversation-Tracker](https://github.com/cmbarinaga/Conversation-Tracker), branch `main`.

## What is complete

The local prototype implements Steps 1–3 and has been run on the AMI meetings ES2002a (374 sentences), ES2002b (749), and ES2002c (721): 1,844 sentences total. It downloads real transcripts, preserves their order, embeds each sentence with all-MiniLM-L6-v2, fits PCA per meeting, and writes a self-contained browser GUI.

The GUI offers future-aware animated thread, original causal thread, and original PCA comparison. Playback includes pause/resume, restart/replay, speed, fractional scrubbing, tip-following camera, smooth fixed Bézier segments, and end-of-conversation fallback.

Step 3 prepares actual-tokenizer context passages: 128 past tokens and 24/48/96 future tokens, default 48. An additional future budget can be selected at generation with `--future-tokens`. Past and future alignment remain separate. The full turning heuristic and simulated input buffering are documented in the README.

## Baseline evidence and limitations

- 17 Python tests and 14 Node tests passed on the original Windows machine.
- All 7,364 context windows were independently recounted with the actual MiniLM tokenizer, checking budgets and current-sentence exclusion.
- All three meetings and their causal/PCA comparisons were inspected in a browser. Buffering, pause, replay, EOF fallback, and mode switching were verified.
- At default settings, the pricing continuation (ES2002a 216) changed from 60.0° to 14.6°, and the RSI expansion (ES2002b 273) from 57.7° to 15.0°.
- The genuine slogan shift (ES2002c 199) also changed from 57.5° to 19.4°. This reduces over-turning; it does not demonstrate reliable topic-change classification. Left/right and loops have no validated semantic interpretation.

`sample-output/` preserves the baseline outputs and screenshots, so the cloud agent can inspect results before recomputing. `sample-output/conversation_tracker.html` opens offline; serve that folder for a browser preview when needed. Generated files normally go to ignored `output/`.

## Set up the cloud environment

In Codex, choose **Work in > Cloud > Select environment > Create environment**, select this GitHub repo, and choose **Get started**. Give the setup agent the following project-specific instructions:

> Read AGENTS.md and CLOUD_HANDOFF.md. Prepare Python 3.12, uv, and Node 22 or newer. Run `bash scripts/setup-cloud.sh`, which installs locked CPU dependencies, runs both test suites, and generates the real-data Step 3 viewer without opening a desktop browser. Configure a browser preview for the generated output when needed. Preserve the existing causal and PCA comparisons. Report setup/test results before publishing the environment.

Allow the package-manager downloads plus the AMI server (`groups.inf.ed.ac.uk`), CPU PyTorch (`download.pytorch.org`), and Hugging Face model downloads (`huggingface.co` and its redirected model/CDN hosts). Python provisioning through uv may also use GitHub-hosted downloads. No API key, paid inference service, or private dataset is required. If a network policy blocks a redirect, add the specific reported download destination and retry; do not disable TLS verification.

Review the setup report, then **Publish**. After **Environment published**, choose **Start a new task**. A useful first task is:

> Read AGENTS.md, README.md, and CLOUD_HANDOFF.md. Continue from the completed Step 3 baseline. Confirm the environment is ready and summarize the remaining turn-calibration limitations before making further changes.

Official workflow: [OpenAI cloud environment documentation](https://learn.chatgpt.com/docs/environments/cloud-environments). The cloud environment still needs to be created and published in the user's account; pushing this repository does not create one automatically. The new cloud agent receives the checked-in handoff, not this chat's full history.

## Commands

```bash
bash scripts/setup-cloud.sh

# Later checks and regeneration:
uv run --frozen --python 3.12 python -m unittest discover -s tests -v
node --test tests/thread_math.test.js
uv run --frozen --python 3.12 python conversation_tracker.py --no-open

# Optional preview using the cloud environment's preview/port access:
uv run --python 3.12 python -m http.server 8765 --bind 0.0.0.0 --directory output
```

Use the cloud environment's port access rather than a local Windows filesystem path. Stop preview servers when finished. Review the task's diff and tests, then commit or open a pull request to keep subsequent work in GitHub. Linux execution must be verified during cloud setup; the original validation was on Windows.
