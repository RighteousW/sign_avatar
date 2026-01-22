"""
Extract top 1000 translations by BLEU-4 score from gesture vocab dataset
"""

import torch
import csv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

from ..constants import GLOSS2TEXT_MODEL_SYNTHETIC
from ..model_training.gloss2text import load_full_model, translate_sentence

# Configuration
DATA_PATH = "data/dataset/synthetic/synthetic_NSL_gesture_vocab.csv"
output_path = "demo_sentences.txt"


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load model
    print("Loading model...")
    model, gloss_vocab, text_vocab, config = load_full_model(GLOSS2TEXT_MODEL_SYNTHETIC, device)
    model.eval()
    print("Model loaded!\n")

    # Load data
    print("Loading dataset...")
    data = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gloss_text = row.get("gloss", row.get("Gloss", ""))
            text_text = row.get("text", row.get("Text", ""))
            if gloss_text and text_text:
                data.append(
                    {
                        "gloss": gloss_text.strip().split(),
                        "reference": text_text.strip().split(),
                    }
                )
    print(f"Loaded {len(data)} samples\n")

    # Evaluate all samples
    print("Evaluating translations...")
    smoothing = SmoothingFunction().method1
    results = []

    for item in tqdm(data, desc="Translating"):
        gloss = item["gloss"]
        reference = item["reference"]

        # Translate
        translation = translate_sentence(model, gloss, gloss_vocab, text_vocab, device)

        # Calculate BLEU score (adapt weights based on sentence length)
        if translation and reference:
            # Use appropriate n-gram weights based on sentence length
            min_len = min(len(reference), len(translation))

            if min_len >= 4:
                # Standard BLEU-4
                weights = (0.25, 0.25, 0.25, 0.25)
            elif min_len == 3:
                # BLEU-3
                weights = (0.33, 0.33, 0.33, 0)
            elif min_len == 2:
                # BLEU-2
                weights = (0.5, 0.5, 0, 0)
            else:
                # BLEU-1
                weights = (1, 0, 0, 0)

            bleu_score = sentence_bleu(
                [reference], translation, weights=weights, smoothing_function=smoothing
            )

            results.append(
                {
                    "bleu4": bleu_score,
                    "gloss": " ".join(gloss),
                    "reference": " ".join(reference),
                    "translation": " ".join(translation),
                    "length": min_len,
                }
            )

    # Sort by BLEU score (descending), then by length (descending)
    results.sort(key=lambda x: (x["bleu4"], x["length"]), reverse=True)

    # Take top 1000
    top_1000 = results[:1000]

    # Save to file
    print(f"\nSaving top 1000 translations to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TOP 1000 TRANSLATIONS BY BLEU-4 SCORE\n")
        f.write("=" * 80 + "\n\n")

        for i, item in enumerate(top_1000, 1):
            f.write(
                f"Rank #{i} | Score: {item['bleu4']:.4f} | Length: {item['length']}\n"
            )
            f.write(f"Gloss:       {item['gloss']}\n")
            f.write(f"Reference:   {item['reference']}\n")
            f.write(f"Translation: {item['translation']}\n")
            f.write("-" * 80 + "\n\n")

    # Print statistics
    print("\nStatistics:")
    print(f"  Total evaluated: {len(results)}")
    print(f"  Top 1000 saved to: {output_path}")
    print(
        f"  Average BLEU-4 (top 1000): {sum(x['bleu4'] for x in top_1000) / 1000:.4f}"
    )
    print(f"  Best BLEU-4: {top_1000[0]['bleu4']:.4f}")
    print(f"  1000th BLEU-4: {top_1000[-1]['bleu4']:.4f}")

    # Show top 5
    print("\nTop 5 translations:")
    print("=" * 80)
    for i, item in enumerate(top_1000[:5], 1):
        print(f"\n#{i} | Score: {item['bleu4']:.4f} | Length: {item['length']}")
        print(f"Gloss:       {item['gloss']}")
        print(f"Reference:   {item['reference']}")
        print(f"Translation: {item['translation']}")

    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
