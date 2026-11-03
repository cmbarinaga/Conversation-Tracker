# Conversation Tracker — Step 2 Handover

## Goal

Extend the working Step 1 prototype into a **dynamic, animated conversation thread** that reflects semantic continuity and divergence, rather than absolute embedding positions.

Reuse the existing sentence segmentation, embedding model, PCA projection, and GUI.

## Experiment 1: Contextual Alignment

For each sentence embedding \(\mathbf z_i\), calculate its similarity to the previous \(K=5\) sentences using a recency-weighted context embedding:

\[
\mathbf c_i=\operatorname{normalize}\left(\sum_{k=1}^{K}w_k\mathbf z_{i-k}\right)
\]

\[
A_i=\cos(\mathbf z_i,\mathbf c_i)
\]

High alignment indicates semantic continuity; low alignment indicates divergence.

Make \(K\) adjustable in the GUI.

## Experiment 2: Relative Semantic Trajectory

Instead of plotting absolute PCA coordinates, construct an incremental trajectory.

**Turning direction:** Project consecutive embedding differences onto the existing PCA basis:

\[
\mathbf d_i=P(\mathbf z_i-\mathbf z_{i-1})
\]

Determine left/right turning direction using the 2D cross product between \(\mathbf d_i\) and the current heading \(\mathbf h_{i-1}\):

\[
s_i=\operatorname{sign}(h_xd_y-h_yd_x)
\]

If the direction is ambiguous, continue straight.

**Turning magnitude:** Use contextual divergence:

\[
D_i=\operatorname{clip}(1-A_i,0,1)
\]

\[
\Delta\phi_i=s_i\theta_{\max}D_i
\]

Update heading and position:

\[
\phi_i=\phi_{i-1}+\Delta\phi_i
\]

\[
\mathbf p_i=\mathbf p_{i-1}+\ell
\begin{bmatrix}\cos\phi_i\\\sin\phi_i\end{bmatrix}
\]

Initially use constant segment length \(\ell\). Make \(\theta_{\max}\) adjustable.

## Animation Requirements

**Do not produce only a static plot.**

- Animate the thread growing sentence by sentence, tracing each segment progressively.
- Use smooth cubic Bézier curves, with control points derived from consecutive headings, to avoid sharp corners.
- Keep previously drawn segments fixed as new sentences arrive.
- Show a moving tip and the currently animated sentence.
- Implement play, pause, restart, speed adjustment, and timeline scrubbing.
- Have the viewport smoothly follow the moving tip.
- Use a minimal dark GUI with a thin teal thread.

Preserve the original PCA plot as an alternative view for comparison.

## Completion Criteria

1. Contextual alignment and relative trajectory are implemented.
2. The animated thread curves smoothly according to semantic divergence.
3. All playback controls work.
4. Three sample conversations demonstrate continuity, gradual topic changes, and abrupt changes.
5. The application runs end-to-end without errors.

## Instructions to Codex

Implement, run, inspect, and iterate until the working animated prototype is complete. Make reasonable decisions independently; do not stop to report unfinished work unless genuinely blocked.

**Do not implement backchannel filtering, LLMs, Android integration, extensive evaluation, or unrelated infrastructure.**

Deliver the working GUI and exact launch command.