import itertools
import csv

from ..audio2gloss import AudioToGlossConverter

def load_first_n_sentences(filepath, n_lines=100_000):
    """
    Loads and strips the first N non-empty lines (sentences) from a text file.
    """
    print(f"Loading first {n_lines} lines from {filepath}...")
    with open(filepath, "r", encoding="utf-8-sig") as f:  # Handle BOM characters
        lines_to_read = itertools.islice(f, n_lines)
        sentences = [line.strip() for line in lines_to_read if line.strip()]
    print(f"Loaded {len(sentences)} non-empty sentences.")
    return sentences

def load_text_from_csv(filepath, max_sentences=100_000):
    """
    Load English text from ASL CSV file (extracts text column only).
    """
    print(f"Loading text from {filepath}...")
    text_sequences = []
    
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Try different possible column names
            text = row.get("text", row.get("Text", row.get("english", ""))).strip()
            if text:
                text_sequences.append(text)
                if i + 1 >= max_sentences:
                    break
    
    print(f"Loaded {len(text_sequences)} sentences from CSV")
    return text_sequences

def main():
    # Initialize the converter
    nsl_converter = AudioToGlossConverter()
    nsl_converter.load_model()
    MAX_DATASET_SIZE = 100_000

    # Configuration
    HQ_ENGLISH_FILE = "data/dataset/high quality english sentences/train.txt"
    # MEDITOD_FILE = "data/dataset/MediTOD/utterances.csv"
    # ASL_CSV_FILE = "data/dataset/ASLG-PC12 dataset/train.csv"

    text_sequences_1 = load_first_n_sentences(HQ_ENGLISH_FILE, n_lines=2_000_000)
    # text_sequences_2 = load_text_from_csv(ASL_CSV_FILE)
    # text_sequences_3 = load_text_from_csv(MEDITOD_FILE)
    combined_sentences = text_sequences_1
    OUTPUT_FILE = f"data/dataset/synthetic/synthetic_HQ-English-Sentences.csv"

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["gloss", "text"])

        valid_count = 0
        conversion_errors = 0

        for original_text in combined_sentences:
            try:
                number_of_words = len(original_text.split(" "))
                if number_of_words > 20 or number_of_words < 3:
                    continue
                glosses = nsl_converter.text_to_glosses(original_text)

                # Skip empty results
                if not glosses:
                    continue

                # Write valid data
                csv_writer.writerow([glosses, original_text])
                valid_count += 1
                if valid_count >= MAX_DATASET_SIZE:
                    break

            except Exception as e:
                if conversion_errors < 10:  # Only print first few errors
                    print(f"Error processing sentence: {original_text}: {e}")
                conversion_errors += 1
                continue

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Valid samples generated: {valid_count:,}")
    print(f"Conversion errors: {conversion_errors:,}")
    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
