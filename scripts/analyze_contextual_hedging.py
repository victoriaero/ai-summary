"""Shared implementation helpers for the contextual BioScope hedge model."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Token
from spacy.util import filter_spans

from analyze_epistemic_commitment import (
    DEFAULT_BIOSCOPE_DIR,
    DEFAULT_COLLECTIONS_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPACY_MODEL,
    build_hedge_matcher,
    discover_responses,
    extract_bioscope_lexicon,
    metadata_subset,
    normalize_cue,
)


SEED = 20260930
MODEL_VERSION = "bioscope-context-logreg-v1"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "contextual_hedging"


def flatten_xml(element) -> tuple[str, list[tuple[int, int, str]]]:
    pieces: list[str] = []
    spans: list[tuple[int, int, str]] = []

    def visit(node) -> None:
        if node.text:
            pieces.append(node.text)
        for child in node:
            start = sum(map(len, pieces))
            visit(child)
            end = sum(map(len, pieces))
            if child.tag == "cue" and child.attrib.get("type", "").casefold() == "speculation":
                surface = "".join(child.itertext())
                if surface.strip():
                    spans.append((start, end, surface))
            if child.tail:
                pieces.append(child.tail)

    visit(element)
    return "".join(pieces), spans


def bioscope_sentences(bioscope_dir: Path) -> list[dict[str, Any]]:
    paths = [bioscope_dir / "abstracts.xml", bioscope_dir / "full_papers.xml",
             bioscope_dir / "clinical_merger" / "clinical_records_anon.xml"]
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        root = ET.parse(path).getroot()
        for doc_index, document in enumerate(root.iter("Document")):
            doc_id = next((x.text or "" for x in document.findall("DocID")), str(doc_index))
            group = f"{path.name}:{doc_id}"
            for sentence in document.iter("sentence"):
                text, cues = flatten_xml(sentence)
                if text.strip():
                    rows.append({"document_id": group, "sentence_id": sentence.attrib.get("id", ""), "text": text, "cues": cues})
    if not rows:
        raise ValueError("No annotated sentences found in BioScope XML")
    return rows


def token_features(token: Token) -> dict[str, str | bool]:
    features: dict[str, str | bool] = {
        "bias": True, "lower": token.lower_, "lemma": token.lemma_.casefold(),
        "pos": token.pos_, "tag": token.tag_, "shape": token.shape_,
        "is_title": token.is_title, "is_upper": token.is_upper,
        "is_punct": token.is_punct, "suffix3": token.text[-3:].casefold(),
        "prefix3": token.text[:3].casefold(),
    }
    for offset in (1, 2):
        for direction, index in (("p", token.i - offset), ("n", token.i + offset)):
            if 0 <= index < len(token.doc):
                neighbor = token.doc[index]
                features[f"{direction}{offset}:lower"] = neighbor.lower_
                features[f"{direction}{offset}:lemma"] = neighbor.lemma_.casefold()
                features[f"{direction}{offset}:pos"] = neighbor.pos_
                features[f"{direction}{offset}:shape"] = neighbor.shape_
            else:
                features[f"{direction}{offset}:BND"] = True
    return features


def labels_for_doc(doc: Doc, spans: list[tuple[int, int, str]]) -> list[int]:
    return [int(any(tok.idx < end and tok.idx + len(tok) > start for start, end, _ in spans)) for tok in doc]


def sentence_data(rows: list[dict[str, Any]], nlp):
    docs = list(nlp.pipe((r["text"] for r in rows), batch_size=128))
    enriched, features, labels, groups = [], [], [], []
    for row, doc in zip(rows, docs):
        y = labels_for_doc(doc, row["cues"])
        enriched.append({**row, "doc": doc, "labels": y})
        features.extend(token_features(tok) for tok in doc)
        labels.extend(y)
        groups.extend([row["document_id"]] * len(doc))
    return enriched, features, labels, groups


def stratified_split(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(groups))
    train, remainder = next(GroupShuffleSplit(n_splits=1, test_size=.30, random_state=SEED).split(indices, groups=groups))
    dev_rel, test_rel = next(GroupShuffleSplit(n_splits=1, test_size=.50, random_state=SEED + 1).split(remainder, groups=groups[remainder]))
    return train, remainder[dev_rel], remainder[test_rel]


def contiguous_spans(doc: Doc, probabilities: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    positive = probabilities >= threshold
    spans, start = [], None
    for i, is_positive in enumerate(list(positive) + [False]):
        if is_positive and start is None:
            start = i
        elif not is_positive and start is not None:
            span = doc[start:i]
            spans.append({"text": span.text, "start": span.start_char, "end": span.end_char,
                          "probability": float(np.mean(probabilities[start:i]))})
            start = None
    return spans


def exact_span_metrics(rows, threshold: float) -> dict[str, float]:
    tp = fp = fn = 0
    for row in rows:
        gold = {(a, b) for a, b, _ in row["cues"]}
        pred = {(x["start"], x["end"]) for x in contiguous_spans(row["doc"], row["probabilities"], threshold)}
        tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def choose_threshold(rows, model) -> float:
    for row in rows:
        row["probabilities"] = model.predict_proba([token_features(t) for t in row["doc"]])[:, 1] if len(row["doc"]) else np.array([])
    scores = [(exact_span_metrics(rows, float(t))["f1"], float(t)) for t in np.arange(.10, .91, .05)]
    return max(scores, key=lambda pair: (pair[0], -abs(pair[1] - .5)))[1]


def evaluate_partition(rows, model, threshold: float) -> dict[str, Any]:
    gold_sentence, predicted_sentence, all_gold, all_pred = [], [], [], []
    for row in rows:
        probs = model.predict_proba([token_features(t) for t in row["doc"]])[:, 1] if len(row["doc"]) else np.array([])
        row["probabilities"] = probs
        pred = probs >= threshold
        all_gold.extend(row["labels"]); all_pred.extend(pred.astype(int).tolist())
        gold_sentence.append(int(bool(row["cues"]))); predicted_sentence.append(int(bool(pred.any())))
    p, r, f, _ = precision_recall_fscore_support(all_gold, all_pred, average="binary", zero_division=0)
    span = exact_span_metrics(rows, threshold)
    return {"token_precision": float(p), "token_recall": float(r), "token_f1": float(f),
            "cue_span_precision": span["precision"], "cue_span_recall": span["recall"], "cue_span_f1": span["f1"],
            "sentence_accuracy": float(accuracy_score(gold_sentence, predicted_sentence)),
            "sentence_f1": float(f1_score(gold_sentence, predicted_sentence, zero_division=0)), "n_sentences": len(rows)}


def collect_response_predictions(responses: pd.DataFrame, nlp, model, threshold: float):
    occurrence_rows, metric_rows, lexical_candidates = [], [], []
    lexicon, _ = extract_bioscope_lexicon(DEFAULT_BIOSCOPE_DIR)
    matcher = build_hedge_matcher(nlp, lexicon)
    for (_, response), doc in zip(responses.iterrows(), nlp.pipe(responses["aio_text"].tolist(), batch_size=64)):
        probs = model.predict_proba([token_features(t) for t in doc])[:, 1] if len(doc) else np.array([])
        spans = contiguous_spans(doc, probs, threshold)
        sentences = list(doc.sents)
        sentence_by_token = {t.i: (i + 1, sent) for i, sent in enumerate(sentences) for t in sent}
        hedged_sentences = set()
        for span in spans:
            token = next((t for t in doc if t.idx <= span["start"] < t.idx + len(t)), None)
            if token is None: continue
            sentence_id, sentence = sentence_by_token[token.i]
            hedged_sentences.add(sentence_id)
            occurrence_rows.append({**metadata_subset(response), "sentence_id": sentence_id, "sentence_text": sentence.text,
                "cue_surface": span["text"], "cue_start": span["start"], "cue_end": span["end"],
                "cue_probability": span["probability"], "detected_as_hedge": True, "model_version": MODEL_VERSION})
        n_tokens = sum(not t.is_space and not t.is_punct for t in doc)
        metric_rows.append({**metadata_subset(response), "n_sentences": len(sentences), "n_tokens": n_tokens,
            "n_contextual_hedges": len(spans), "n_hedged_sentences": len(hedged_sentences),
            "contextual_hedges_per_100_tokens": 100 * len(spans) / n_tokens if n_tokens else np.nan,
            "contextual_hedged_sentence_rate": len(hedged_sentences) / len(sentences) if sentences else np.nan,
            "model_version": MODEL_VERSION})
        matches = [doc[a:b] for _, a, b in matcher(doc)]
        for candidate in filter_spans(matches):
            sentence_id, sentence = sentence_by_token[candidate.start]
            candidate_probs = probs[candidate.start:candidate.end]
            predicted = bool(len(candidate_probs) and np.any(candidate_probs >= threshold))
            cue = normalize_cue(candidate.text)
            freq = lexicon.loc[lexicon.normalized_cue.eq(cue), "frequency_in_resource"]
            lexical_candidates.append({**metadata_subset(response), "sentence_id": sentence_id, "sentence_text": sentence.text,
                "highlighted_candidate": candidate.text, "candidate_start": candidate.start_char, "candidate_end": candidate.end_char,
                "candidate_frequency": int(freq.max()) if len(freq) else 0, "model_prediction": "yes" if predicted else "no"})
    return pd.DataFrame(occurrence_rows), pd.DataFrame(metric_rows), lexical_candidates


def make_blind_sample(candidates: list[dict[str, Any]], output_dir: Path) -> None:
    df = pd.DataFrame(candidates)
    columns = ["annotation_id", "response_id", "sentence_id", "sentence_text", "highlighted_candidate",
               "human_label_1", "human_label_2", "adjudicated_label"]
    if df.empty:
        pd.DataFrame(columns=columns).to_csv(output_dir / "human_validation_blind.csv", index=False)
        return
    median_frequency = df.candidate_frequency.median()
    df["cue_frequency_band"] = np.where(df.candidate_frequency >= median_frequency, "frequent", "rare")
    df["prediction_stratum"] = df.model_prediction.map({"yes": "predicted_hedge", "no": "predicted_non_hedge"})
    factors = ["group_type", "dimension", "replica_id", "cue_frequency_band", "prediction_stratum"]
    df["stratum"] = df[factors].fillna("").astype(str).agg("|".join, axis=1)
    target = min(400, len(df)); counts = df.stratum.value_counts()
    allocation = (counts / counts.sum() * target).round().astype(int)
    allocation[allocation.eq(0)] = 1
    while allocation.sum() > target:
        can_reduce = allocation[allocation > 1]
        if can_reduce.empty: break
        allocation[can_reduce.idxmax()] -= 1
    while allocation.sum() < target:
        remain = counts - allocation
        remain = remain[remain > 0]
        if remain.empty: break
        allocation[remain.idxmax()] += 1
    picks = []
    for key, n in allocation.items():
        part = df.loc[df.stratum.eq(key)]
        picks.extend(part.sample(n=min(int(n), len(part)), random_state=SEED).index.tolist())
    sample = df.loc[sorted(set(picks))].copy()
    sample["annotation_id"] = [f"H{i:04d}" for i in range(1, len(sample) + 1)]
    for col in ("human_label_1", "human_label_2", "adjudicated_label"):
        sample[col] = ""
    sample[columns].to_csv(output_dir / "human_validation_blind.csv", index=False)
    sample[["annotation_id", "response_id", "candidate_start", "candidate_end", "model_prediction", "cue_frequency_band", "stratum"]].to_csv(output_dir / "human_validation_key.csv", index=False)
