"""
Simple test script for NSL Gloss Converter with BLEU evaluation
"""

import csv
from collections import Counter
import math

from src.audio2gloss.nsl_gloss_converter import NSLGlossConverter


def calculate_bleu(reference, candidate, n):
    """Calculate BLEU-n score"""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    if len(cand_tokens) < n:
        return 0.0

    # Get n-grams
    ref_ngrams = Counter(
        [tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)]
    )
    cand_ngrams = Counter(
        [tuple(cand_tokens[i : i + n]) for i in range(len(cand_tokens) - n + 1)]
    )

    # Clipped precision
    clipped = sum(min(cand_ngrams[ng], ref_ngrams[ng]) for ng in cand_ngrams)
    total = sum(cand_ngrams.values())

    if total == 0:
        return 0.0

    precision = clipped / total

    # Brevity penalty
    bp = (
        1.0
        if len(cand_tokens) > len(ref_tokens)
        else math.exp(1 - len(ref_tokens) / len(cand_tokens))
    )

    return bp * precision


# Load converter
converter = NSLGlossConverter()
converter.load_model()

# Load utterances
utterances = []
with open("data/dataset/MediTOD/utterances.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        utterances.append(row["text"])

# Test first 100
utterances = utterances[:100]

# Calculate scores
bleu1_total = 0
bleu2_total = 0
bleu3_total = 0
bleu4_total = 0

for i, text in enumerate(utterances):
    # Text -> Gloss -> Text
    gloss = converter.text_to_gloss(text)
    gloss_str = " ".join(gloss)
    reconstructed = converter.gloss_to_text(gloss_str)

    # Calculate BLEU
    bleu1 = calculate_bleu(text, reconstructed, 1)
    bleu2 = calculate_bleu(text, reconstructed, 2)
    bleu3 = calculate_bleu(text, reconstructed, 3)
    bleu4 = calculate_bleu(text, reconstructed, 4)

    bleu1_total += bleu1
    bleu2_total += bleu2
    bleu3_total += bleu3
    bleu4_total += bleu4

    if i < 5:  # Show first 5 examples
        print(f"\nOriginal: {text}")
        print(f"Gloss: {gloss_str}")
        print(f"Reconstructed: {reconstructed}")

# Print results
n = len(utterances)
print(f"\n--- Results on {n} samples ---")
print(f"BLEU-1: {bleu1_total/n:.4f}")
print(f"BLEU-2: {bleu2_total/n:.4f}")
print(f"BLEU-3: {bleu3_total/n:.4f}")
print(f"BLEU-4: {bleu4_total/n:.4f}")
