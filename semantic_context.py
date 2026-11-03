"""Token-budgeted passage context using the embedding model's own tokenizer."""
from functools import lru_cache


def checked_encode(model, texts, *, label="passages"):
    """Fail explicitly before SentenceTransformer can silently truncate anything."""
    counts = [len(ids) for ids in model.tokenizer(
        texts, add_special_tokens=True, truncation=False,
    )["input_ids"]]
    limit = int(model.max_seq_length)
    oversized = [(i + 1, count) for i, count in enumerate(counts) if count > limit]
    if oversized:
        index, count = oversized[0]
        raise ValueError(f"{label} item {index} has {count} model tokens; maximum is {limit}. "
                         "Shorten the utterance or context budget; nothing was silently truncated.")
    import numpy as np
    embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=True)
    if embeddings.shape != (len(texts), 384) or not np.isfinite(embeddings).all():
        raise ValueError(f"{label}: model returned invalid embeddings.")
    return embeddings


def context_windows(texts, tokenizer, budget, direction):
    """Nearest past/future content, trimmed only at the far token boundary.

    Budgets include special tokens. Exports keep source indices and the exact text
    actually embedded, so model input lengths and lookahead releases are auditable.
    """
    special = tokenizer.num_special_tokens_to_add(pair=False)
    if budget <= special or direction not in {"past", "future"}:
        raise ValueError("Context budget must leave room for content; direction is past or future.")

    @lru_cache(maxsize=None)
    def tokens(text):
        return tokenizer(text, add_special_tokens=False, truncation=False,
                         return_offsets_mapping=True)

    result = []
    for current in range(len(texts)):
        neighbors = range(current - 1, -1, -1) if direction == "past" else range(current + 1, len(texts))
        selected = []
        for index in neighbors:
            selected.append(index)
            ordered = sorted(selected)
            passage = " ".join(texts[j] for j in ordered)
            if len(tokens(passage)["input_ids"]) + special >= budget:
                break
        if not selected:
            result.append(dict(text="", tokens=0, first=None, last=None))
            continue
        ordered = sorted(selected)
        passage = " ".join(texts[j] for j in ordered)
        start, end = 0, len(passage)
        offsets = tokens(passage)["offset_mapping"]
        capacity = budget - special
        if len(offsets) > capacity:
            if direction == "past":
                start = offsets[-capacity][0]
            else:
                end = offsets[capacity - 1][1]
        # A cut within a WordPiece can change its tokenization. Recount the exact
        # substring and advance the far boundary until it really fits.
        while len(tokens(passage[start:end])["input_ids"]) + special > budget:
            trimmed = tokens(passage[start:end])["offset_mapping"]
            if direction == "past":
                start += trimmed[0][1]
            else:
                end = start + trimmed[-1][0]
        included = []
        cursor = 0
        for index in ordered:
            sentence_end = cursor + len(texts[index])
            if sentence_end > start and cursor < end:
                included.append(index)
            cursor = sentence_end + 1
        text = passage[start:end].strip()
        count = len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        if count > budget:
            raise ValueError("Context token budget was exceeded after assembly.")
        result.append(dict(text=text, tokens=count, first=included[0], last=included[-1]))
    return result


def build_context_profiles(model, texts, sentence_embeddings, past_budget=128, future_budgets=(24, 48, 96)):
    import numpy as np

    limit = int(model.max_seq_length)
    special = model.tokenizer.num_special_tokens_to_add(pair=False)
    if any(not special < b <= limit for b in (past_budget, *future_budgets)):
        raise ValueError(f"Token budgets must be between {special + 1} and model maximum {limit}, inclusive.")
    past = context_windows(texts, model.tokenizer, past_budget, "past")
    futures = {str(b): context_windows(texts, model.tokenizer, b, "future") for b in future_budgets}
    passages = list(dict.fromkeys(w["text"] for windows in [past, *futures.values()]
                                 for w in windows if w["text"]))
    if passages:
        print(f"Embedding {len(passages)} distinct context passages...", flush=True)
        vectors = checked_encode(model, passages, label="context passage")
        lookup = dict(zip(passages, vectors, strict=True))
    else:
        lookup = {}

    def alignment(windows):
        return [float(np.clip(np.dot(sentence_embeddings[i], lookup[w["text"]]), -1, 1))
                if w["text"] else None for i, w in enumerate(windows)]

    profiles = {}
    past_alignment = alignment(past)
    for budget, future in futures.items():
        profiles[budget] = dict(
            pastAlignment=past_alignment, futureAlignment=alignment(future),
            pastTokens=[w["tokens"] for w in past], futureTokens=[w["tokens"] for w in future],
            # A segment cannot be released until its final future utterance arrives.
            readyAt=[w["last"] if w["last"] is not None else i for i, w in enumerate(future)],
        )
    audit = dict(modelMaxTokens=limit, pastBudget=past_budget,
                 futureBudgets=list(future_budgets), past=past, future=futures, profiles=profiles)
    return dict(pastBudget=past_budget, profiles=profiles), audit
