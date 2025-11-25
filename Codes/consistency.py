# src/llm_consistency/int_consist.py

from __future__ import annotations

import itertools
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .config import MODEL_CONFIG
from .client import run_conversation, ConversationResult, MessageStats

# External metric libs
import sacrebleu
from rouge_score import rouge_scorer

# Readability
try:
    import textstat

    _HAS_TEXTSTAT = True
except ImportError:  # pragma: no cover
    _HAS_TEXTSTAT = False

# TF-IDF
try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

# POS tagging
try:
    import nltk
    from nltk import pos_tag, word_tokenize

    _HAS_NLTK = True
except ImportError:  # pragma: no cover
    _HAS_NLTK = False

# Progress bar
try:
    from tqdm.auto import tqdm

    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# Basic text helpers
# ---------------------------------------------------------------------------


def _simple_tokenize(text: str) -> List[str]:
    """Very simple whitespace tokenizer."""
    return [t for t in text.strip().split() if t]


def _sentences(text: str) -> List[str]:
    """Naive sentence splitter."""
    import re

    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def _ngram_set(tokens: List[str], n: int = 2) -> set:
    if len(tokens) < n:
        return set()
    return set(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _jaccard(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return 1.0
    union = s1 | s2
    if not union:
        return 0.0
    return len(s1 & s2) / len(union)


# ---------------------------------------------------------------------------
# Style & POS helpers
# ---------------------------------------------------------------------------


def _style_features(text: str) -> Tuple[float, float]:
    """
    Style features:
      - type-token ratio (TTR)
      - average sentence length in tokens
    """
    tokens = _simple_tokenize(text)
    if not tokens:
        return 0.0, 0.0

    ttr = len(set(tokens)) / len(tokens)
    sents = _sentences(text)
    if sents:
        avg_sent_len = len(tokens) / len(sents)
    else:
        avg_sent_len = float(len(tokens))

    return ttr, avg_sent_len


def _style_similarity(t1: str, t2: str) -> float:
    """Convert style feature distance into (0, 1] similarity."""
    ttr1, asl1 = _style_features(t1)
    ttr2, asl2 = _style_features(t2)
    dist = math.sqrt((ttr1 - ttr2) ** 2 + (asl1 - asl2) ** 2)
    return 1.0 / (1.0 + dist)


def _pos_distribution(text: str) -> Dict[str, float]:
    """
    Coarse POS distribution: NOUN, VERB, ADJ, ADV, OTHER.
    Uses NLTK if available; otherwise returns a flat distribution.
    """
    if not _HAS_NLTK:
        # Fallback: pretend we don't know, return equal mass.
        return {"NOUN": 0.2, "VERB": 0.2, "ADJ": 0.2, "ADV": 0.2, "OTHER": 0.2}

    tokens = word_tokenize(text)
    if not tokens:
        return {"NOUN": 0.0, "VERB": 0.0, "ADJ": 0.0, "ADV": 0.0, "OTHER": 0.0}

    tags = pos_tag(tokens)

    counts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0, "OTHER": 0}

    for _, tag in tags:
        if tag.startswith("N"):  # NN, NNP, etc.
            counts["NOUN"] += 1
        elif tag.startswith("V"):  # VB, VBD, etc.
            counts["VERB"] += 1
        elif tag.startswith("J"):  # JJ, JJR, etc.
            counts["ADJ"] += 1
        elif tag.startswith("R"):  # RB, RBR, etc.
            counts["ADV"] += 1
        else:
            counts["OTHER"] += 1

    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}

    return {k: v / total for k, v in counts.items()}


def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    keys = set(v1.keys()) | set(v2.keys())
    a = [v1.get(k, 0.0) for k in keys]
    b = [v2.get(k, 0.0) for k in keys]

    dot = sum(x * y for x, y in zip(a, b))
    norm1 = math.sqrt(sum(x * x for x in a))
    norm2 = math.sqrt(sum(y * y for y in b))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# ---------------------------------------------------------------------------
# Metric helpers (internal / reference)
# ---------------------------------------------------------------------------


def _mean_pairwise(
    texts: List[str], sim_fn: Callable[[str, str], float]
) -> Optional[float]:
    """Mean pairwise similarity metric over all (i, j), i < j."""
    if len(texts) < 2:
        return None
    vals: List[float] = []
    for i, j in itertools.combinations(range(len(texts)), 2):
        vals.append(sim_fn(texts[i], texts[j]))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _mean_vs_reference(
    texts: List[str],
    reference: Optional[str],
    sim_fn: Callable[[str, str], float],
) -> Optional[float]:
    """Mean similarity vs a single reference text."""
    if reference is None or reference == "":
        return None
    if not texts:
        return None
    vals = [sim_fn(t, reference) for t in texts]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _ngram_jaccard_bigram(t1: str, t2: str) -> float:
    tokens1 = _simple_tokenize(t1)
    tokens2 = _simple_tokenize(t2)
    return _jaccard(_ngram_set(tokens1, 2), _ngram_set(tokens2, 2))


def _tfidf_cosine_internal(texts: List[str]) -> Optional[float]:
    if not _HAS_SKLEARN:
        return None
    if len(texts) < 2:
        return None

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)  # [n_runs, vocab]

    vals: List[float] = []
    for i, j in itertools.combinations(range(len(texts)), 2):
        v1 = X[i]
        v2 = X[j]
        # rows are L2-normalized by default, so dot product = cosine
        sim_mat = v1 @ v2.T  # 1x1 sparse matrix
        sim = sim_mat.toarray()[0, 0]
        vals.append(float(sim))

    if not vals:
        return None
    return sum(vals) / len(vals)


def _tfidf_cosine_reference(
    texts: List[str], reference: Optional[str]
) -> Optional[float]:
    if not _HAS_SKLEARN:
        return None
    if reference is None or reference == "":
        return None
    if not texts:
        return None

    corpus = [reference] + texts
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    ref_vec = X[0]
    vals: List[float] = []
    for i in range(1, X.shape[0]):
        sim_mat = ref_vec @ X[i].T  # 1x1 sparse matrix
        sim = sim_mat.toarray()[0, 0]
        vals.append(float(sim))

    if not vals:
        return None
    return sum(vals) / len(vals)


def _bleu_internal(texts: List[str]) -> Optional[float]:
    if len(texts) < 2:
        return None
    vals: List[float] = []
    for i, j in itertools.combinations(range(len(texts)), 2):
        hyp = [texts[i]]
        ref = [[texts[j]]]
        score = sacrebleu.corpus_bleu(hyp, ref).score  # 0..100
        vals.append(score)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _bleu_reference(texts: List[str], reference: Optional[str]) -> Optional[float]:
    if reference is None or reference == "":
        return None
    if not texts:
        return None

    refs = [[reference] * len(texts)]
    score = sacrebleu.corpus_bleu(texts, refs).score  # 0..100
    return score


_ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _rougeL_f1(t1: str, t2: str) -> float:
    """ROUGE-L F1 between t1 (reference) and t2 (hypothesis)."""
    scores = _ROUGE_SCORER.score(t1, t2)
    return float(scores["rougeL"].fmeasure)


def _rougeL_internal(texts: List[str]) -> Optional[float]:
    if len(texts) < 2:
        return None
    vals: List[float] = []
    for i, j in itertools.combinations(range(len(texts)), 2):
        vals.append(_rougeL_f1(texts[i], texts[j]))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _rougeL_reference(texts: List[str], reference: Optional[str]) -> Optional[float]:
    if reference is None or reference == "":
        return None
    if not texts:
        return None
    vals = [_rougeL_f1(reference, t) for t in texts]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _pos_similarity_internal(texts: List[str]) -> Optional[float]:
    if len(texts) < 2:
        return None

    dists = [_pos_distribution(t) for t in texts]
    vals: List[float] = []
    for i, j in itertools.combinations(range(len(dists)), 2):
        vals.append(_cosine(dists[i], dists[j]))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _pos_similarity_reference(
    texts: List[str], reference: Optional[str]
) -> Optional[float]:
    if reference is None or reference == "":
        return None
    if not texts:
        return None

    ref_dist = _pos_distribution(reference)
    vals = [_cosine(ref_dist, _pos_distribution(t)) for t in texts]
    if not vals:
        return None
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Readability & basic statistics
# ---------------------------------------------------------------------------


def _readability(text: str) -> Tuple[float, float]:
    """
    Returns (flesch_reading_ease, flesch_kincaid_grade).
    If textstat is not available, returns (nan, nan).
    """
    if not _HAS_TEXTSTAT or not text.strip():
        return float("nan"), float("nan")

    try:
        fre = float(textstat.flesch_reading_ease(text))
        fkg = float(textstat.flesch_kincaid_grade(text))
    except Exception:
        fre, fkg = float("nan"), float("nan")
    return fre, fkg


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


# ---------------------------------------------------------------------------
# Confidence helpers (log-probs)
# ---------------------------------------------------------------------------


def _average_logprob_for_messages(msgs: List[MessageStats]) -> Optional[float]:
    all_lps: List[float] = []
    for msg in msgs:
        if msg.token_logprobs:
            all_lps.extend(t.logprob for t in msg.token_logprobs)
    if not all_lps:
        return None
    return sum(all_lps) / len(all_lps)


def _average_logprob_for_message(msg: MessageStats) -> Optional[float]:
    if not msg.token_logprobs:
        return None
    vals = [t.logprob for t in msg.token_logprobs]
    if not vals:
        return None
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Main internal consistency function
# ---------------------------------------------------------------------------


def int_consist(
    model: str,
    user_prompts: List[str],
    system_prompt: Optional[str] = None,
    reference: Optional[List[str]] = None,
    n_runs: int = 5,
    api_key: Optional[str] = None,
    request_logprobs_default: bool = True,
    show_progress: bool = True,
    **gen_kwargs: Any,
) -> pd.DataFrame:
    """
    Internal consistency of a model over multiple runs.

    Arguments
    ---------
    model:
        Name of the model in MODEL_CONFIG (e.g. "gpt-4.1").
    user_prompts:
        List of user prompts (one assistant message per prompt).
    system_prompt:
        Optional system prompt.
    reference:
        Optional list of reference texts, one per user prompt.
        If provided, len(reference) must equal len(user_prompts).
        They are used as fixed anchors for similarity metrics.
    n_runs:
        Number of independent runs (conversations) to sample.
    api_key:
        Optional API key (otherwise resolved via environment).
    request_logprobs_default:
        If True and the model supports logprobs, request them from the client.
    show_progress:
        If True, show a tqdm progress bar (if tqdm is installed).
    **gen_kwargs:
        Additional generation kwargs (temperature, max_tokens, etc.).

    Output
    ------
    pandas.DataFrame with:
        columns: "total", "m1", "m2", ..., "mN"
        rows (example subset):
            - length_chars_mean, length_chars_std
            - length_tokens_mean, length_tokens_std
            - num_sentences_mean, num_sentences_std
            - ttr_mean, ttr_std
            - avg_sentence_length_mean, avg_sentence_length_std
            - readability_flesch_mean, readability_flesch_std
            - readability_kincaid_mean, readability_kincaid_std
            - tfidf_cosine_internal, tfidf_cosine_reference
            - ngram_jaccard_bigram_internal, ngram_jaccard_bigram_reference
            - bleu_internal, bleu_reference
            - rougeL_internal, rougeL_reference
            - style_similarity_internal, style_similarity_reference
            - pos_distribution_similarity_internal, pos_distribution_similarity_reference
            - confidence_avg_logprob_mean, confidence_avg_logprob_std   (if supported)
    """
    if not user_prompts:
        raise ValueError("user_prompts must contain at least one prompt string.")

    num_msgs = len(user_prompts)

    if reference is not None and len(reference) != num_msgs:
        raise ValueError(
            "If provided, `reference` must have the same length as `user_prompts`."
        )

    cfg = MODEL_CONFIG.get(model)
    if cfg is None:
        raise ValueError(f"Unknown model '{model}'. Check MODEL_CONFIG in config.py.")

    supports_logprobs = bool(cfg.get("supports_logprobs", False))
    request_logprobs = (
        [request_logprobs_default]
        if supports_logprobs and request_logprobs_default
        else [False]
    )

    # Segment labels: total conversation + each assistant message m1..mN
    segment_labels = ["total"] + [f"m{i+1}" for i in range(num_msgs)]

    # Segment -> list of texts (one per run)
    seg_texts: Dict[str, List[str]] = {seg: [] for seg in segment_labels}
    # Segment -> list of avg logprobs (one per run) if available
    seg_conf_values: Dict[str, List[float]] = {seg: [] for seg in segment_labels}
    # Segment -> single reference text (if provided)
    seg_reference: Dict[str, Optional[str]] = {seg: None for seg in segment_labels}

    if reference is not None:
        seg_reference["total"] = "\n".join(reference)
        for i, ref_msg in enumerate(reference):
            seg_reference[f"m{i+1}"] = ref_msg

    # -----------------------------------------------------------------------
    # Run model n_runs times and collect texts + confidences
    # -----------------------------------------------------------------------
    if show_progress and _HAS_TQDM and n_runs > 1:
        run_iter = tqdm(
            range(n_runs),
            desc=f"int_consist {model}",
            unit="run",
        )
    else:
        run_iter = range(n_runs)

    for _ in run_iter:
        conv: ConversationResult = run_conversation(
            model_name=model,
            user_prompts=user_prompts,
            api_key=api_key,
            system_prompt=system_prompt,
            request_logprobs=request_logprobs,
            **gen_kwargs,
        )

        assistant_msgs: List[MessageStats] = conv.assistant_messages

        # Texts
        total_text = "\n".join(msg.content for msg in assistant_msgs)
        seg_texts["total"].append(total_text)

        for i in range(num_msgs):
            if i < len(assistant_msgs):
                seg_texts[f"m{i+1}"].append(assistant_msgs[i].content)
            else:
                seg_texts[f"m{i+1}"].append("")

        # Confidence
        if supports_logprobs and request_logprobs_default:
            total_lp = _average_logprob_for_messages(assistant_msgs)
            if total_lp is not None:
                seg_conf_values["total"].append(total_lp)

            for i in range(num_msgs):
                if i < len(assistant_msgs):
                    lp = _average_logprob_for_message(assistant_msgs[i])
                    if lp is not None:
                        seg_conf_values[f"m{i+1}"].append(lp)

    # -----------------------------------------------------------------------
    # Build metrics for each segment
    # -----------------------------------------------------------------------

    # Prepare metric rows
    rows: Dict[str, Dict[str, float]] = {
        # Basic text statistics
        "length_chars_mean": {},
        "length_chars_std": {},
        "length_tokens_mean": {},
        "length_tokens_std": {},
        "num_sentences_mean": {},
        "num_sentences_std": {},
        "ttr_mean": {},
        "ttr_std": {},
        "avg_sentence_length_mean": {},
        "avg_sentence_length_std": {},
        "readability_flesch_mean": {},
        "readability_flesch_std": {},
        "readability_kincaid_mean": {},
        "readability_kincaid_std": {},
        # TF-IDF & n-gram overlap
        "tfidf_cosine_internal": {},
        "tfidf_cosine_reference": {},
        "ngram_jaccard_bigram_internal": {},
        "ngram_jaccard_bigram_reference": {},
        # BLEU / ROUGE
        "bleu_internal": {},
        "bleu_reference": {},
        "rougeL_internal": {},
        "rougeL_reference": {},
        # Style & POS
        "style_similarity_internal": {},
        "style_similarity_reference": {},
        "pos_distribution_similarity_internal": {},
        "pos_distribution_similarity_reference": {},
    }

    # Confidence rows only if we have any data
    any_conf_data = any(len(vals) > 0 for vals in seg_conf_values.values())
    if supports_logprobs and any_conf_data:
        rows["confidence_avg_logprob_mean"] = {}
        rows["confidence_avg_logprob_std"] = {}

    # Fill rows per segment
    for seg in segment_labels:
        texts = seg_texts[seg]
        ref_text = seg_reference[seg]

        # --- Per-run basic stats ---
        length_chars: List[float] = []
        length_tokens: List[float] = []
        num_sentences: List[float] = []
        ttr_list: List[float] = []
        avg_sent_len_list: List[float] = []
        fre_list: List[float] = []
        fkg_list: List[float] = []

        for t in texts:
            length_chars.append(float(len(t)))
            tokens = _simple_tokenize(t)
            length_tokens.append(float(len(tokens)))

            sents = _sentences(t)
            num_sentences.append(float(len(sents)))

            ttr, asl = _style_features(t)
            ttr_list.append(ttr)
            avg_sent_len_list.append(asl)

            fre, fkg = _readability(t)
            fre_list.append(fre)
            fkg_list.append(fkg)

        # Means & stds
        (
            rows["length_chars_mean"][seg],
            rows["length_chars_std"][seg],
        ) = _mean_std(length_chars)
        (
            rows["length_tokens_mean"][seg],
            rows["length_tokens_std"][seg],
        ) = _mean_std(length_tokens)
        (
            rows["num_sentences_mean"][seg],
            rows["num_sentences_std"][seg],
        ) = _mean_std(num_sentences)
        rows["ttr_mean"][seg], rows["ttr_std"][seg] = _mean_std(ttr_list)
        (
            rows["avg_sentence_length_mean"][seg],
            rows["avg_sentence_length_std"][seg],
        ) = _mean_std(avg_sent_len_list)
        (
            rows["readability_flesch_mean"][seg],
            rows["readability_flesch_std"][seg],
        ) = _mean_std(fre_list)
        (
            rows["readability_kincaid_mean"][seg],
            rows["readability_kincaid_std"][seg],
        ) = _mean_std(fkg_list)

        # --- TF-IDF & n-gram overlap ---
        tfidf_int = _tfidf_cosine_internal(texts)
        rows["tfidf_cosine_internal"][seg] = (
            float(tfidf_int) if tfidf_int is not None else float("nan")
        )

        tfidf_ref = _tfidf_cosine_reference(texts, ref_text)
        rows["tfidf_cosine_reference"][seg] = (
            float(tfidf_ref) if tfidf_ref is not None else float("nan")
        )

        ngram_int = _mean_pairwise(texts, _ngram_jaccard_bigram)
        rows["ngram_jaccard_bigram_internal"][seg] = (
            float(ngram_int) if ngram_int is not None else float("nan")
        )

        ngram_ref = _mean_vs_reference(texts, ref_text, _ngram_jaccard_bigram)
        rows["ngram_jaccard_bigram_reference"][seg] = (
            float(ngram_ref) if ngram_ref is not None else float("nan")
        )

        # --- BLEU / ROUGE ---
        bleu_int = _bleu_internal(texts)
        rows["bleu_internal"][seg] = (
            float(bleu_int) if bleu_int is not None else float("nan")
        )

        bleu_ref = _bleu_reference(texts, ref_text)
        rows["bleu_reference"][seg] = (
            float(bleu_ref) if bleu_ref is not None else float("nan")
        )

        rouge_int = _rougeL_internal(texts)
        rows["rougeL_internal"][seg] = (
            float(rouge_int) if rouge_int is not None else float("nan")
        )

        rouge_ref = _rougeL_reference(texts, ref_text)
        rows["rougeL_reference"][seg] = (
            float(rouge_ref) if rouge_ref is not None else float("nan")
        )

        # --- Style & POS ---
        style_int = _mean_pairwise(texts, _style_similarity)
        rows["style_similarity_internal"][seg] = (
            float(style_int) if style_int is not None else float("nan")
        )

        style_ref = _mean_vs_reference(texts, ref_text, _style_similarity)
        rows["style_similarity_reference"][seg] = (
            float(style_ref) if style_ref is not None else float("nan")
        )

        pos_int = _pos_similarity_internal(texts)
        rows["pos_distribution_similarity_internal"][seg] = (
            float(pos_int) if pos_int is not None else float("nan")
        )

        pos_ref = _pos_similarity_reference(texts, ref_text)
        rows["pos_distribution_similarity_reference"][seg] = (
            float(pos_ref) if pos_ref is not None else float("nan")
        )

        # --- Confidence ---
        if supports_logprobs and any_conf_data:
            vals = seg_conf_values[seg]
            mean_lp, std_lp = _mean_std(vals)
            rows["confidence_avg_logprob_mean"][seg] = mean_lp
            rows["confidence_avg_logprob_std"][seg] = std_lp

    df = pd.DataFrame.from_dict(rows, orient="index", columns=segment_labels)
    return df
