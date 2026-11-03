# Conversation Tracker — Codex Handover

## Goal
Build a minimal desktop prototype that visualizes a written conversation as a continuous path through 2D semantic space. **Only implement Step 1.** Do not build the Android app or add speech recognition yet.

## Step 1 — Four tasks

1. **Get sample conversations.** Download a small set of publicly available conversation transcripts from a reputable dataset of your choosing (for example, the AMI Meeting Corpus). Use real, ordered dialogue rather than unrelated standalone sentences. Follow the dataset's license and document the source.
2. **Segment into sentences.** Choose a reasonable sentence-splitting method. Preserve the original order and text of each sentence. Handle basic transcript formatting as needed.
3. **Embed the sentences.** Use a pretrained sentence embedding model, preferably `sentence-transformers/all-MiniLM-L6-v2`. Produce one embedding per sentence. Project those embeddings into 2D using PCA initially.
4. **Visualize the trajectory.** Build a minimal desktop GUI (Plotly in a local browser is fine). Plot each sentence at its 2D coordinate and connect points **in conversational order**, not by spatial proximity. Show the sentence text on hover. Use a simple dark background and a clearly visible line.

## Definition of done
- A single documented command launches or generates the visualization using downloaded sample conversation data.
- At least one real conversation is rendered as an ordered 2D trajectory.
- Hovering a point reveals its sentence.
- The code and a short README explain how to run it.

## Codex working instructions
Implement the four tasks end to end. Choose reasonable libraries and make decisions independently. Run the pipeline and fix errors until it works. Do not stop to report intermediate progress unless a genuine blocker requires user input. Avoid additional infrastructure, elaborate metrics, large evaluation suites, model quantization, or premature optimization.

## Later (not part of Step 1)
Once the user has inspected and approved the visualization, integrate on-device speech-to-text (Sherpa-ONNX streaming Zipformer) and the embedding model, test the combined pipeline on a computer, then port it to a Pixel 8 Pro Android app. The eventual Android UI should be an immersive full-screen trajectory with tap-to-pause. Speaker identification and user-facing metrics are out of scope for V1.