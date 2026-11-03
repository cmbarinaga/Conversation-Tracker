# Source and processing

Credit: AMI Project / AMI Meeting Corpus, manual annotations v1.6.2.

Source: https://groups.inf.ed.ac.uk/ami/download/

Archive: https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip

License: Creative Commons Attribution 4.0 International (CC BY 4.0). See AMI-LICENCE.txt and https://creativecommons.org/licenses/by/4.0/.

Meetings used: ES2002a, ES2002b, ES2002c. These are recorded scenario-based design meetings.

Changes: nonverbal XML annotations omitted; original spoken tokens reconstructed with punctuation spacing; segments split at annotated sentence-ending punctuation; trailing fragments retained. Sentences ordered by onset timestamp, then source speaker ID and token position for ties. Overlapping speech is ordered by onset.

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (Apache 2.0), normalized 384-dimensional vectors. Each meeting has its own two-component PCA fit; axes are not comparable across meetings.

Animated thread: recency-weighted context (weights 0.8^(lag-1), default K=5); cosine alignment determines turn magnitude, and PCA coordinate differences determine left/right direction. Constant endpoint step length 1; default maximum turn 60 degrees. Curves use heading-derived cubic Bezier control points. This is an experimental visualization, not a validated topic-change detector. All sentences, including backchannels, are retained.

Step 3 default: future-aware passage alignment using the model tokenizer, 128 past / 48 future model tokens (including special tokens). Budgets may be changed at launch. See each meeting's contexts.json for exact embedded text, counts, separate scores, and release indices. Departure is modulated by forward support, then smoothed one-sided; scores are not averaged. Playback simulates one incoming sentence per second at 1x and waits for required future speech. The original causal thread and PCA remain available.
