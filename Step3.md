# Conversation Tracker — Step 3 Handover

## Goal

Improve the existing animated semantic trajectory using **token-based context windows** and **non-causal (future-aware) alignment**.

Reuse the existing embedding model, PCA projection, and animation. Do not rebuild the application.

## 1. Token-Based Context Windows

Replace fixed sentence-count context windows with token-budgeted windows.

- **Past context:** Collect preceding utterances backward until reaching approximately 128 model tokens.
- **Future context:** Collect subsequent utterances forward until reaching approximately 48 model tokens.
- Use the embedding model's actual tokenizer to count tokens.
- Embed each assembled context as a passage, rather than averaging individual sentence embeddings.
- Preserve individual sentence embeddings for trajectory generation.
- Prevent silent truncation at the embedding model's maximum input length.

This should prevent short acknowledgments from crowding substantive sentences out of the context.

## 2. Non-Causal Contextual Alignment

For each current sentence \(u_i\), calculate:

\[
A_i^-=\operatorname{sim}_{\cos}(E(u_i),E(\text{past context}))
\]

\[
A_i^+=\operatorname{sim}_{\cos}(E(u_i),E(\text{future context}))
\]

Interpret these jointly:

- High past / high future: continuing a topic.
- Low past / high future: initiating a new direction.
- High past / low future: potentially ending a topic.
- Low past / low future: possible isolated tangent.

**Do not simply average the two scores**, since that can conceal genuine topic transitions.

Use both measurements to influence the trajectory: past alignment controls semantic departure, while future alignment helps distinguish sustained transitions from isolated deviations.

Keep the turning direction derived from the existing PCA displacement method.

## 3. Delayed Animation

Buffer sufficient future speech before finalizing the current segment.

- Introduce a configurable lookahead delay based on the future token budget.
- Use future context to shape smooth, sustained turns rather than isolated sharp bends.
- Once drawn, previous segments must remain fixed.
- At the end of a conversation, gracefully fall back to available context.

Keep existing playback controls and the original causal visualization available for comparison.

## Completion Criteria

Implement, run, and inspect the updated animated trajectory on the existing sample conversations. Confirm token budgets, non-causal alignment, smooth animation, and deterministic replay work.

Make reasonable implementation decisions independently. Do not add LLMs, rule-based filler filtering, Android integration, or extensive validation infrastructure.

Deliver the working GUI and exact launch command.