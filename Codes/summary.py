# src/summary.py

from __future__ import annotations

from typing import Optional

import math
import pandas as pd


def _get(df: pd.DataFrame, row: str, col: str) -> Optional[float]:
    """Safe getter: returns None if row/col missing or NaN."""
    if row not in df.index:
        return None
    if col not in df.columns:
        return None
    val = df.loc[row, col]
    try:
        f = float(val)
    except Exception:
        return None
    if math.isnan(f):
        return None
    return f


def summarize_int_consist(df: pd.DataFrame, segments: Optional[List[str]] = None) -> str:
    """
    Produce a human-readable summary of an internal-consistency DataFrame
    produced by int_consist().

    By default, summarizes *all* segments (columns): "total", "m1", "m2", ...

    You can also restrict to a subset:
        summarize_int_consist(df, segments=["total", "m1", "m2"])
    """
    if segments is None:
        # use all columns in order
        segments = list(df.columns)

    lines: list[str] = []

    for col in segments:
        if col not in df.columns:
            continue  # silently skip unknown columns

        lines.append(f"INTERNAL CONSISTENCY SUMMARY ({col})")
        lines.append("-" * (len(lines[-1])))

        # --- length / style ---
        len_tok_mean = _get(df, "length_tokens_mean", col)
        len_tok_std = _get(df, "length_tokens_std", col)
        ttr_mean = _get(df, "ttr_mean", col)
        ttr_std = _get(df, "ttr_std", col)
        asl_mean = _get(df, "avg_sentence_length_mean", col)
        asl_std = _get(df, "avg_sentence_length_std", col)
        fre_mean = _get(df, "readability_flesch_mean", col)
        fkg_mean = _get(df, "readability_kincaid_mean", col)

        if len_tok_mean is not None and len_tok_std is not None:
            lines.append(
                f"- Length: ~{len_tok_mean:.1f} tokens on average "
                f"(std ≈ {len_tok_std:.1f} across runs)."
            )

        if ttr_mean is not None and ttr_std is not None:
            lines.append(
                f"- Type–Token Ratio (lexical diversity): mean ≈ {ttr_mean:.3f}, "
                f"std ≈ {ttr_std:.3f}."
            )

        if asl_mean is not None and asl_std is not None:
            lines.append(
                f"- Average sentence length: ≈ {asl_mean:.1f} tokens per sentence "
                f"(std ≈ {asl_std:.1f})."
            )

        if fre_mean is not None and fkg_mean is not None:
            lines.append(
                f"- Readability (Flesch ≈ {fre_mean:.1f}, "
                f"Flesch–Kincaid grade ≈ {fkg_mean:.1f})."
            )

        # --- lexical / semantic consistency ---
        tfidf_int = _get(df, "tfidf_cosine_internal", col)
        ngram_int = _get(df, "ngram_jaccard_bigram_internal", col)
        bleu_int = _get(df, "bleu_internal", col)
        rouge_int = _get(df, "rougeL_internal", col)
        style_int = _get(df, "style_similarity_internal", col)
        pos_int = _get(df, "pos_distribution_similarity_internal", col)

        lines.append("")
        lines.append("Lexical / semantic stability across runs:")

        if tfidf_int is not None:
            lines.append(f"- TF-IDF cosine similarity: {tfidf_int:.3f} (1.0 = identical).")

        if ngram_int is not None:
            lines.append(
                f"- Bigram Jaccard overlap: {ngram_int:.3f} "
                "(similar to ROUGE-2; 1.0 = identical n-grams)."
            )

        if bleu_int is not None:
            lines.append(f"- BLEU (pairwise mean): {bleu_int:.1f} (0–100 scale).")

        if rouge_int is not None:
            lines.append(f"- ROUGE-L F1 (pairwise mean): {rouge_int:.3f}.")

        if style_int is not None:
            lines.append(
                f"- Style similarity (TTR + sentence length): {style_int:.3f} "
                "(1.0 = very similar style)."
            )

        if pos_int is not None:
            lines.append(
                f"- POS distribution similarity: {pos_int:.3f} "
                "(1.0 = identical part-of-speech mix)."
            )

        # --- reference-based metrics ---
        tfidf_ref = _get(df, "tfidf_cosine_reference", col)
        ngram_ref = _get(df, "ngram_jaccard_bigram_reference", col)
        bleu_ref = _get(df, "bleu_reference", col)
        rouge_ref = _get(df, "rougeL_reference", col)
        style_ref = _get(df, "style_similarity_reference", col)
        pos_ref = _get(df, "pos_distribution_similarity_reference", col)

        if any(
            v is not None
            for v in [tfidf_ref, ngram_ref, bleu_ref, rouge_ref, style_ref, pos_ref]
        ):
            lines.append("")
            lines.append("Similarity to reference (if provided):")

            if tfidf_ref is not None:
                lines.append(f"- TF-IDF cosine vs reference: {tfidf_ref:.3f}.")
            if ngram_ref is not None:
                lines.append(f"- Bigram Jaccard vs reference: {ngram_ref:.3f}.")
            if bleu_ref is not None:
                lines.append(f"- BLEU vs reference: {bleu_ref:.1f} (0–100 scale).")
            if rouge_ref is not None:
                lines.append(f"- ROUGE-L F1 vs reference: {rouge_ref:.3f}.")
            if style_ref is not None:
                lines.append(f"- Style similarity vs reference: {style_ref:.3f}.")
            if pos_ref is not None:
                lines.append(f"- POS distribution similarity vs reference: {pos_ref:.3f}.")

        # --- confidence (logprobs) ---
        conf_mean = _get(df, "confidence_avg_logprob_mean", col)
        conf_std = _get(df, "confidence_avg_logprob_std", col)
        if conf_mean is not None and conf_std is not None:
            lines.append("")
            lines.append("Confidence (only if logprobs available):")
            lines.append(
                f"- Avg token log-prob: mean ≈ {conf_mean:.3f}, "
                f"std ≈ {conf_std:.3f} "
                "(higher = more confident, more stable)."
            )

        # blank line between segments
        lines.append("")
        lines.append("")

    return "\n".join(lines)



def summarize_ext_consist(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
) -> str:
    """
    Produce a short human-readable summary of an external-consistency
    DataFrame produced by ext_consist().

    Focuses on the 'total' column and highlights:
      - which model is more internally stable
      - how similar the two models are to each other
      - how each model aligns with the reference (if any)
      - confidence differences if logprobs available
    """
    col = "total"
    lines: list[str] = []

    lines.append(f"EXTERNAL CONSISTENCY SUMMARY ({model_a} vs {model_b})")
    lines.append("----------------------------------------------------")

    # --- internal stability per model (TF-IDF, BLEU, ROUGE) ---
    tfidf_a = _get(df, "tfidf_cosine_internal_model_a", col)
    tfidf_b = _get(df, "tfidf_cosine_internal_model_b", col)
    bleu_a = _get(df, "bleu_internal_model_a", col)
    bleu_b = _get(df, "bleu_internal_model_b", col)
    rouge_a = _get(df, "rougeL_internal_model_a", col)
    rouge_b = _get(df, "rougeL_internal_model_b", col)

    lines.append("Internal stability (each model vs itself across runs):")

    if tfidf_a is not None and tfidf_b is not None:
        better = model_a if tfidf_a >= tfidf_b else model_b
        lines.append(
            f"- TF-IDF cosine internal: {model_a} ≈ {tfidf_a:.3f}, "
            f"{model_b} ≈ {tfidf_b:.3f} → {better} is slightly more lexically stable."
        )
    else:
        if tfidf_a is not None:
            lines.append(f"- {model_a} TF-IDF internal: {tfidf_a:.3f}.")
        if tfidf_b is not None:
            lines.append(f"- {model_b} TF-IDF internal: {tfidf_b:.3f}.")

    if bleu_a is not None and bleu_b is not None:
        lines.append(
            f"- BLEU internal (0–100): {model_a} ≈ {bleu_a:.1f}, "
            f"{model_b} ≈ {bleu_b:.1f}."
        )

    if rouge_a is not None and rouge_b is not None:
        lines.append(
            f"- ROUGE-L internal: {model_a} ≈ {rouge_a:.3f}, "
            f"{model_b} ≈ {rouge_b:.3f}."
        )

    # --- cross-model similarity ---
    tfidf_between = _get(df, "tfidf_cosine_between_models", col)
    ngram_between = _get(df, "ngram_jaccard_bigram_between_models", col)
    bleu_between = _get(df, "bleu_between_models", col)
    rouge_between = _get(df, "rougeL_between_models", col)
    style_between = _get(df, "style_similarity_between_models", col)
    pos_between = _get(df, "pos_distribution_similarity_between_models", col)

    lines.append("")
    lines.append("Similarity between models (outputs compared directly):")

    if tfidf_between is not None:
        lines.append(
            f"- TF-IDF cosine between models: {tfidf_between:.3f} "
            "(1.0 = very similar bag-of-words content)."
        )
    if ngram_between is not None:
        lines.append(
            f"- Bigram Jaccard between models: {ngram_between:.3f} "
            "(1.0 = very similar phrasing)."
        )
    if bleu_between is not None:
        lines.append(
            f"- BLEU between models (0–100): {bleu_between:.1f} "
            "(higher = more similar wording)."
        )
    if rouge_between is not None:
        lines.append(
            f"- ROUGE-L between models: {rouge_between:.3f} "
            "(higher = more similar sequence of tokens)."
        )
    if style_between is not None:
        lines.append(
            f"- Style similarity between models: {style_between:.3f} "
            "(1.0 = very similar style)."
        )
    if pos_between is not None:
        lines.append(
            f"- POS distribution similarity between models: {pos_between:.3f} "
            "(1.0 = very similar grammatical structure)."
        )

    # --- reference alignment per model ---
    tfidf_ref_a = _get(df, "tfidf_cosine_reference_model_a", col)
    tfidf_ref_b = _get(df, "tfidf_cosine_reference_model_b", col)
    bleu_ref_a = _get(df, "bleu_reference_model_a", col)
    bleu_ref_b = _get(df, "bleu_reference_model_b", col)
    rouge_ref_a = _get(df, "rougeL_reference_model_a", col)
    rouge_ref_b = _get(df, "rougeL_reference_model_b", col)

    if any(v is not None for v in [tfidf_ref_a, tfidf_ref_b, bleu_ref_a, bleu_ref_b, rouge_ref_a, rouge_ref_b]):
        lines.append("")
        lines.append("Alignment with reference (if provided):")
        if tfidf_ref_a is not None and tfidf_ref_b is not None:
            better = model_a if tfidf_ref_a >= tfidf_ref_b else model_b
            lines.append(
                f"- TF-IDF vs reference: {model_a} ≈ {tfidf_ref_a:.3f}, "
                f"{model_b} ≈ {tfidf_ref_b:.3f} → {better} closer in bag-of-words space."
            )
        if bleu_ref_a is not None and bleu_ref_b is not None:
            lines.append(
                f"- BLEU vs reference (0–100): {model_a} ≈ {bleu_ref_a:.1f}, "
                f"{model_b} ≈ {bleu_ref_b:.1f}."
            )
        if rouge_ref_a is not None and rouge_ref_b is not None:
            lines.append(
                f"- ROUGE-L vs reference: {model_a} ≈ {rouge_ref_a:.3f}, "
                f"{model_b} ≈ {rouge_ref_b:.3f}."
            )

    # --- confidence --- (if any logprobs in DF)
    conf_mean_a = _get(df, "confidence_avg_logprob_mean_model_a", col)
    conf_std_a = _get(df, "confidence_avg_logprob_std_model_a", col)
    conf_mean_b = _get(df, "confidence_avg_logprob_mean_model_b", col)
    conf_std_b = _get(df, "confidence_avg_logprob_std_model_b", col)

    if any(v is not None for v in [conf_mean_a, conf_mean_b]):
        lines.append("")
        lines.append("Confidence (avg token log-prob, if available):")
        if conf_mean_a is not None and conf_std_a is not None:
            lines.append(
                f"- {model_a}: mean ≈ {conf_mean_a:.3f}, std ≈ {conf_std_a:.3f}."
            )
        if conf_mean_b is not None and conf_std_b is not None:
            lines.append(
                f"- {model_b}: mean ≈ {conf_mean_b:.3f}, std ≈ {conf_std_b:.3f}."
            )

    return "\n".join(lines)
