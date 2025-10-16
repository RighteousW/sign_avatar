import itertools  # Necessary for efficient slicing
import csv

from ..audio2gloss import AudioToGlossConverter
from ..model_training import load_data_from_file


def load_first_n_sentences(filepath, n_lines=2_000_000):
    """
    Loads and strips the first N non-empty lines (sentences) from a text file.
    """
    print(f"Loading first {n_lines} lines from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        # Use itertools.islice to read only the first N lines of the file
        lines_to_read = itertools.islice(f, n_lines)
        # Process the lines: strip whitespace and filter out lines that become empty
        sentences = [line.strip() for line in lines_to_read if line.strip()]
    print(f"Loaded {len(sentences)} non-empty sentences.")
    return sentences


def main():
    # Initialize the converter
    nsl_converter = AudioToGlossConverter(debug=False)
    nsl_converter.load_model()

    # --- Configuration ---
    INPUT_FILE = "data/dataset/ASLG-PC12 dataset/train.csv"
    OUTPUT_FILE = "data/dataset/ASLG-PC12 dataset/synthetic.csv"

    _, text_sequences = load_data_from_file(INPUT_FILE, "csv")

    # --- Conversion and Saving ---
    print(f"Starting conversion and saving to {OUTPUT_FILE}...")

    # Open the new file for writing with 'newline'="" for proper CSV handling
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        # Create a CSV writer object
        csv_writer = csv.writer(f)

        # Write the header row
        csv_writer.writerow(["gloss", "text"])

        # Process each sentence
        for i, original_text in enumerate(text_sequences):
            if (i + 1) % 10000 == 0:
                print(f"Processing sentence {i + 1}...")

            clause_glosses_list = nsl_converter.text_to_glosses(" ".join((original_text)))
            flattened_gloss_list = [
                gloss for clause in clause_glosses_list for gloss in clause
            ]

            # Format the flattened glosses into a single space-separated string
            gloss_sequence = " ".join(flattened_gloss_list)

            # Write the gloss and the original text as a new row in CSV format
            csv_writer.writerow([gloss_sequence, original_text])

    print(
        f"Successfully generated synthetic gloss data for {len(text_sequences)} sentences and saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
