"""Download AMI transcripts and render an animated semantic thread plus PCA comparison."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass, replace
import html
import json
from pathlib import Path
import re
import shutil
import sys
import textwrap
import urllib.request
import webbrowser
import xml.etree.ElementTree as ET
import zipfile
from semantic_context import build_context_profiles, checked_encode


ROOT = Path(__file__).resolve().parent
ARCHIVE_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/"
    "ami_public_manual_1.6.2.zip"
)
SOURCE_URL = "https://groups.inf.ed.ac.uk/ami/download/"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MEETINGS = ("ES2002a", "ES2002b", "ES2002c")
NITE = "{http://nite.sourceforge.net/}"


@dataclass(frozen=True)
class Sentence:
    meeting: str
    speaker: str  # Source metadata only; not a speaker-identification feature.
    source_position: int
    start: float
    end: float
    text: str
    index: int = 0


def download_archive(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "ami_public_manual_1.6.2.zip"
    if destination.exists() and zipfile.is_zipfile(destination):
        print("Using cached AMI annotations.", flush=True)
        return destination
    partial = destination.with_suffix(".zip.part")
    print("Downloading official AMI annotations (about 22 MB)...", flush=True)
    try:
        request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "ConversationTracker/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target)
        if not zipfile.is_zipfile(partial):
            raise ValueError("The AMI download was not a ZIP archive.")
        partial.replace(destination)
    except (OSError, ValueError) as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download AMI annotations from {ARCHIVE_URL}. "
            "Check your internet connection and rerun the command. "
            f"Details: {exc}"
        ) from exc
    return destination


def reconstruct_text(tokens: list[str]) -> str:
    """Keep token spelling; remove spaces only around transcript punctuation."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return text.strip()


def parse_speaker(words_xml: bytes, segments_xml: bytes, meeting: str, speaker: str) -> list[Sentence]:
    """Resolve NITE word ranges, then split on annotated terminal punctuation."""
    words = list(ET.fromstring(words_xml))
    positions = {word.attrib[NITE + "id"]: i for i, word in enumerate(words)}
    sentences = []
    for segment in ET.fromstring(segments_xml):
        if segment.tag.rsplit("}", 1)[-1] != "segment":
            continue
        segment_words = []
        for child in segment:
            if child.tag != NITE + "child":
                continue
            ids = re.findall(r"id\(([^)]+)\)", child.attrib["href"])
            if len(ids) not in (1, 2):
                raise ValueError(f"Unsupported word reference: {child.attrib['href']}")
            first, last = positions[ids[0]], positions[ids[-1]]
            if first > last:
                raise ValueError("AMI segment has a reversed word range.")
            segment_words.extend(enumerate(words[first:last + 1], start=first))

        current = []

        def emit() -> None:
            if not current or not any(w.attrib.get("punc") != "true" for _, w in current):
                current.clear()
                return
            first_position, first_word = current[0]
            text = reconstruct_text([w.text.strip() for _, w in current])
            sentences.append(Sentence(
                meeting=meeting, speaker=speaker, source_position=first_position,
                start=float(first_word.attrib.get("starttime", segment.attrib["transcriber_start"])),
                end=float(current[-1][1].attrib.get("endtime", segment.attrib["transcriber_end"])),
                text=text,
            ))
            current.clear()

        for position, word in segment_words:
            if word.tag.rsplit("}", 1)[-1] != "w" or not (word.text or "").strip():
                continue  # Laughter, breaths, and other nonverbal annotations.
            current.append((position, word))
            if word.attrib.get("punc") == "true" and word.text.strip() in {".", "!", "?"}:
                emit()
        emit()  # Preserve incomplete utterances at segment boundaries.
    return sentences


def order_sentences(sentences: list[Sentence]) -> list[Sentence]:
    ordered = sorted(sentences, key=lambda s: (s.start, s.speaker, s.source_position))
    return [replace(sentence, index=i) for i, sentence in enumerate(ordered, start=1)]


def read_meeting(archive: zipfile.ZipFile, meeting: str) -> list[Sentence]:
    result = []
    files = sorted(n for n in archive.namelist() if re.fullmatch(rf"words/{meeting}\.[^.]+\.words\.xml", n))
    if not files:
        raise ValueError(f"No word transcripts found for {meeting}.")
    for name in files:
        speaker = name.split(".")[1]
        segments = f"segments/{meeting}.{speaker}.segments.xml"
        result.extend(parse_speaker(archive.read(name), archive.read(segments), meeting, speaker))
    result = order_sentences(result)
    if len(result) < 2:
        raise ValueError(f"{meeting} contains fewer than two usable sentences.")
    return result


def project_embeddings(embeddings):
    import numpy as np
    from sklearn.decomposition import PCA

    if embeddings.ndim != 2 or embeddings.shape[0] < 2 or embeddings.shape[1] != 384:
        raise ValueError("Expected at least two 384-dimensional sentence embeddings.")
    if not np.isfinite(embeddings).all():
        raise ValueError("The embedding model returned nonfinite values.")
    coordinates = PCA(n_components=2, svd_solver="full").fit_transform(embeddings)
    if not np.isfinite(coordinates).all():
        raise ValueError("PCA returned nonfinite coordinates.")
    return coordinates


def hover_text(sentence: Sentence) -> str:
    # Escape transcript HTML and wrap for readable, complete multiline tooltips.
    return "<br>".join(html.escape(line) for line in textwrap.wrap(
        sentence.text, width=88, break_long_words=False, break_on_hyphens=False,
    ))


def build_figure(results):
    import plotly.graph_objects as go

    figure = go.Figure()
    for meeting_number, (meeting, sentences, coordinates) in enumerate(results):
        visible = meeting_number == 0
        details = [[s.index, hover_text(s)] for s in sentences]
        template = "Sentence %{customdata[0]}<br>%{customdata[1]}<extra></extra>"
        figure.add_trace(go.Scatter(
            x=coordinates[:, 0], y=coordinates[:, 1], mode="lines+markers",
            line=dict(color="#44d9ed", width=2),
            marker=dict(size=6, color="#a5f3fc", line=dict(width=0)),
            customdata=details, hovertemplate=template,
            name=meeting, visible=visible, showlegend=False,
        ))
        for offset, label, color, symbol in (
            (0, "Start", "#c3f584", "diamond"),
            (-1, "End", "#ffba85", "square"),
        ):
            figure.add_trace(go.Scatter(
                x=[coordinates[offset, 0]], y=[coordinates[offset, 1]],
                mode="markers", name=label,
                marker=dict(size=13, color=color, symbol=symbol, line=dict(color="#10151f", width=2)),
                customdata=[details[offset]], hovertemplate=template, visible=visible,
            ))
    buttons = []
    for i, (meeting, _, _) in enumerate(results):
        buttons.append(dict(label=meeting, method="update", args=[
            {"visible": [trace // 3 == i for trace in range(len(results) * 3)]},
            {"title.text": f"Conversation Tracker · {meeting}", "xaxis.autorange": True, "yaxis.autorange": True},
        ]))
    figure.update_layout(
        title=dict(text=f"Conversation Tracker · {results[0][0]}", x=0.035, y=0.97),
        template="plotly_dark", paper_bgcolor="#10151f", plot_bgcolor="#10151f",
        font=dict(family="Segoe UI, Arial, sans-serif", color="#e3ebf5", size=14),
        margin=dict(l=45, r=35, t=125, b=85), autosize=True,
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        hovermode="closest", hoverlabel=dict(bgcolor="#202a3a", font_size=14, align="left"),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.12),
        updatemenus=[dict(type="dropdown", buttons=buttons, x=0, y=1.12,
                         xanchor="left", yanchor="middle", bgcolor="#202a3a", bordercolor="#44546a")],
        annotations=[
            dict(text="Sentences connected in conversation order · Hover to read · Drag to zoom",
                 x=0, y=-0.055, xref="paper", yref="paper", showarrow=False, xanchor="left", font_size=12),
            dict(text=f'<a href="{SOURCE_URL}">AMI Meeting Corpus</a> · CC BY 4.0 · Transcript segmentation and PCA projection',
                 x=0, y=-0.105, xref="paper", yref="paper", showarrow=False, xanchor="left", font_size=11),
        ],
    )
    return figure


def save_source_information(archive: zipfile.ZipFile, output: Path) -> None:
    licenses = [n for n in archive.namelist() if Path(n).name.lower() in {"licence.txt", "license.txt"}]
    if not licenses:
        raise ValueError("The official AMI archive did not contain its expected license.")
    (output / "AMI-LICENCE.txt").write_bytes(archive.read(licenses[0]))
    (output / "SOURCE.md").write_text(
        "# Source and processing\n\n"
        "Credit: AMI Project / AMI Meeting Corpus, manual annotations v1.6.2.\n\n"
        f"Source: {SOURCE_URL}\n\nArchive: {ARCHIVE_URL}\n\n"
        "License: Creative Commons Attribution 4.0 International (CC BY 4.0). "
        "See AMI-LICENCE.txt and https://creativecommons.org/licenses/by/4.0/.\n\n"
        f"Meetings used: {', '.join(MEETINGS)}. These are recorded scenario-based design meetings.\n\n"
        "Changes: nonverbal XML annotations omitted; original spoken tokens reconstructed "
        "with punctuation spacing; segments split at annotated sentence-ending punctuation; "
        "trailing fragments retained. Sentences ordered by onset timestamp, then source "
        "speaker ID and token position for ties. Overlapping speech is ordered by onset.\n\n"
        f"Embedding model: {MODEL} (Apache 2.0), normalized 384-dimensional vectors. "
        "Each meeting has its own two-component PCA fit; axes are not comparable across meetings.\n\n"
        "Animated thread: recency-weighted context (weights 0.8^(lag-1), default K=5); "
        "cosine alignment determines turn magnitude, and PCA coordinate differences determine "
        "left/right direction. Constant endpoint step length 1; default maximum turn 60 degrees. "
        "Curves use heading-derived cubic Bezier control points. This is an experimental "
        "visualization, not a validated topic-change detector. All sentences, including "
        "backchannels, are retained.\n\n"
        "Step 3 default: future-aware passage alignment using the model tokenizer, "
        "128 past / 48 future model tokens (including special tokens). Budgets may be "
        "changed at launch. See each meeting's contexts.json for exact embedded text, "
        "counts, separate scores, and release indices. Departure is modulated by forward "
        "support, then smoothed one-sided; scores are not averaged. Playback simulates "
        "one incoming sentence per second at 1x and waits for required future speech. "
        "The original causal thread and PCA remain available.\n",
        encoding="utf-8",
    )


def build_document(results, embedding_arrays, context_data=None, default_future=48) -> str:
    """Bundle data, Plotly comparison, and browser animation into one offline HTML."""
    import numpy as np
    from plotly.offline import get_plotlyjs

    figure = build_figure(results)
    # Use the shared HTML meeting selector instead of a second Plotly dropdown.
    figure.layout.updatemenus = ()
    figure.update_layout(margin=dict(l=35, r=25, t=65, b=85))
    payload = {"meetings": [], "pca": json.loads(figure.to_json()), "defaultFuture": default_future}
    for (meeting, sentences, coordinates), embeddings in zip(results, embedding_arrays, strict=True):
        if embeddings.shape != (len(sentences), 384) or coordinates.shape != (len(sentences), 2):
            raise ValueError(f"{meeting}: browser data is not aligned with sentences.")
        if not np.isfinite(embeddings).all() or not np.isfinite(coordinates).all():
            raise ValueError(f"{meeting}: browser data contains nonfinite values.")
        entry = {
            "id": meeting, "sentences": [s.text for s in sentences],
            "coordinates": coordinates.tolist(),
            "embeddings": base64.b64encode(np.asarray(embeddings, dtype="<f4").tobytes()).decode("ascii"),
        }
        if context_data is not None:
            entry["context"] = context_data[meeting]
        payload["meetings"].append(entry)
    web = ROOT / "web"
    document = (web / "viewer.html").read_text(encoding="utf-8")
    for marker, source in (
        ("PLOTLY", get_plotlyjs()),
        ("MATH", (web / "thread_math.js").read_text(encoding="utf-8")),
        ("PLAYBACK", (web / "playback.js").read_text(encoding="utf-8")),
        ("VIEWER", (web / "viewer.js").read_text(encoding="utf-8")),
    ):
        document = document.replace(f"@@{marker}@@", source)
    # Transcript content cannot terminate the JSON script element or inject HTML.
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return document.replace("@@DATA@@", encoded)


def run(no_open: bool = False, past_tokens: int = 128, future_tokens: int = 48) -> Path:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    path = download_archive(ROOT / ".cache" / "ami")
    try:
        with zipfile.ZipFile(path) as archive:
            meetings = [(meeting, read_meeting(archive, meeting)) for meeting in MEETINGS]
            save_source_information(archive, output)
    except (ET.ParseError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            f"Could not parse AMI annotations: {exc}. "
            f"Remove {path} and rerun to download a fresh copy."
        ) from exc
    print(f"Loading {MODEL} on CPU (first run downloads model weights)...", flush=True)
    try:
        model = SentenceTransformer(MODEL, device="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {MODEL}. Check internet access to huggingface.co on the first run, "
            f"then retry. No Hugging Face account is required. Details: {exc}"
        ) from exc

    special = model.tokenizer.num_special_tokens_to_add(pair=False)
    if any(not special < budget <= model.max_seq_length for budget in (past_tokens, future_tokens, 24, 48, 96)):
        raise ValueError(f"Token budgets must be between {special + 1} and model maximum {model.max_seq_length}, inclusive.")
    results = []
    embedding_arrays = []
    context_data = {}
    future_budgets = tuple(sorted({24, 48, 96, future_tokens}))
    for meeting, sentences in meetings:
        print(f"Embedding {meeting}: {len(sentences)} sentences...", flush=True)
        texts = [s.text for s in sentences]
        embeddings = checked_encode(model, texts, label=f"{meeting} sentence")
        if embeddings.shape != (len(sentences), 384):
            raise ValueError(f"{meeting}: embedding count does not match sentence count.")
        coordinates = project_embeddings(embeddings)
        (output / f"{meeting}.sentences.json").write_text(
            json.dumps([asdict(s) for s in sentences], ensure_ascii=False, indent=2), encoding="utf-8",
        )
        np.savez_compressed(output / f"{meeting}.npz", embeddings=embeddings, coordinates=coordinates)
        results.append((meeting, sentences, coordinates))
        embedding_arrays.append(embeddings)
        context_data[meeting], audit = build_context_profiles(
            model, texts, embeddings, past_tokens, future_budgets,
        )
        (output / f"{meeting}.contexts.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
        )

    destination = output / "conversation_tracker.html"
    document = build_document(results, embedding_arrays, context_data, future_tokens)
    destination.write_text(document, encoding="utf-8")
    print(f"Visualization: {destination}", flush=True)
    if not no_open and not webbrowser.open(destination.as_uri()):
        print("Browser could not be opened automatically. Open the HTML path above.", flush=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-open", action="store_true", help="Generate files without opening a browser.")
    parser.add_argument("--past-tokens", type=int, default=128, help="Past passage budget including special tokens (default 128).")
    parser.add_argument("--future-tokens", type=int, default=48, help="Default lookahead token budget (default 48).")
    args = parser.parse_args()
    try:
        run(no_open=args.no_open, past_tokens=args.past_tokens, future_tokens=args.future_tokens)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
