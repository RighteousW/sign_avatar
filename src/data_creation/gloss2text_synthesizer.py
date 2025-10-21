import itertools
import csv
import re
from ..audio2gloss import AudioToGlossConverter

def load_first_n_sentences(filepath, n_lines=2_000_000):
    """
    Loads and strips the first N non-empty lines (sentences) from a text file.
    """
    print(f"Loading first {n_lines} lines from {filepath}...")
    with open(filepath, "r", encoding="utf-8-sig") as f:  # Handle BOM characters
        lines_to_read = itertools.islice(f, n_lines)
        sentences = [line.strip() for line in lines_to_read if line.strip()]
    print(f"Loaded {len(sentences)} non-empty sentences.")
    return sentences

def load_text_from_csv(filepath):
    """
    Load English text from ASL CSV file (extracts text column only).
    """
    print(f"Loading text from {filepath}...")
    text_sequences = []
    
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try different possible column names
            text = row.get("text", row.get("Text", row.get("english", ""))).strip()
            if text:
                text_sequences.append(text)
    
    print(f"Loaded {len(text_sequences)} sentences from CSV")
    return text_sequences

def split_long_sentences(text, max_words=25):
    """
    Split long sentences into smaller chunks at natural boundaries.
    
    Args:
        text: Input text
        max_words: Maximum words per sentence
    
    Returns:
        List of shorter sentences
    """
    # First, try splitting on sentence boundaries (. ! ?)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    result = []
    for sent in sentences:
        words = sent.split()
        
        # If sentence is short enough, keep it
        if len(words) <= max_words:
            result.append(sent.strip())
            continue
        
        # For long sentences, try to split on clause boundaries
        # Split on commas, semicolons, colons, "and", "but", "or"
        chunks = re.split(r'[,;:]|\s+(?:and|but|or)\s+', sent, flags=re.IGNORECASE)
        
        current_chunk = []
        for chunk in chunks:
            chunk_words = chunk.strip().split()
            
            # If adding this chunk would exceed limit, save current and start new
            if current_chunk and len(current_chunk) + len(chunk_words) > max_words:
                result.append(' '.join(current_chunk))
                current_chunk = chunk_words
            else:
                current_chunk.extend(chunk_words)
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            # Only add if it's substantive (at least 3 words)
            if len(current_chunk) >= 3:
                result.append(chunk_text)
    
    return result

def validate_sentence(text, min_words=3, max_words=30):
    """
    Check if a sentence is suitable for training.
    
    Args:
        text: Sentence to validate
        min_words: Minimum word count
        max_words: Maximum word count
    
    Returns:
        bool: True if sentence is valid
    """
    words = text.split()
    word_count = len(words)
    
    # Length check
    if word_count < min_words or word_count > max_words:
        return False
    
    # Skip if too many punctuation marks (likely formatting issues)
    punct_ratio = sum(1 for c in text if c in '.,;:!?') / len(text)
    if punct_ratio > 0.15:  # More than 15% punctuation
        return False
    
    # Skip if contains unusual characters (likely corrupted data)
    if any(ord(c) > 127 for c in text if not c.isspace()):
        # Allow common accented characters
        allowed_special = set('áéíóúàèìòùâêîôûäëïöüñçÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÄËÏÖÜÑÇ')
        if any(c not in allowed_special for c in text if ord(c) > 127 and not c.isspace()):
            return False
    
    # Skip if sentence starts with lowercase (likely fragment)
    if text and text[0].islower():
        return False
    
    return True

def main():
    # Initialize the converter
    nsl_converter = AudioToGlossConverter(debug=False)
    nsl_converter.load_model()

    # --- Configuration ---
    ASL_CSV_FILE = "data/dataset/ASLG-PC12 dataset/train.csv"
    OUTPUT_FILE = "data/dataset/ASLG-PC12 dataset/synthetic.csv"

    MAX_SENTENCE_WORDS = 100  # Limit sentence length
    MIN_SENTENCE_WORDS = 1   # Minimum sentence length

    # Load sentences from both sources
    print("\n" + "="*60)
    print("LOADING DATA FROM MULTIPLE SOURCES")
    print("="*60 + "\n")

    text_sequences_2 = load_text_from_csv(ASL_CSV_FILE)

    print(f"\n{'='*60}")
    print(f"RAW SENTENCES LOADED: {len(text_sequences_2):,}")
    print(f"  - ASL CSV sentences: {len(text_sequences_2):,}")
    print(f"{'='*60}\n")

    # # Split and filter sentences
    # print(f"Splitting long sentences (max {MAX_SENTENCE_WORDS} words)...")
    # processed_sentences = []
    # split_count = 0

    # for text in text_sequences_2:
    #     split_sents = split_long_sentences(text, max_words=MAX_SENTENCE_WORDS)

    #     if len(split_sents) > 1:
    #         split_count += 1

    #     for sent in split_sents:
    #         if validate_sentence(sent, min_words=MIN_SENTENCE_WORDS, max_words=MAX_SENTENCE_WORDS):
    #             processed_sentences.append(sent)

    # print(f"Split {split_count:,} long sentences")
    # print(f"After filtering: {len(processed_sentences):,} valid sentences\n")

    processed_sentences = text_sequences_2
    # --- Conversion and Saving ---
    print(f"Starting NSL gloss conversion and saving to {OUTPUT_FILE}...")
    print("="*60 + "\n")

    valid_count = 0
    skipped_count = 0
    conversion_errors = 0
    quality_filtered = 0

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["gloss", "text"])

        for i, original_text in enumerate(processed_sentences):
            if (i + 1) % 10000 == 0:
                print(f"Processing {i + 1:,}/{len(processed_sentences):,} | "
                      f"Valid: {valid_count:,} | "
                      f"Skipped: {skipped_count:,} | "
                      f"Quality filtered: {quality_filtered:,}")

            try:
                clause_glosses_list = nsl_converter.text_to_glosses(original_text)

                # Skip empty results
                if not clause_glosses_list or all(len(clause) == 0 for clause in clause_glosses_list):
                    skipped_count += 1
                    continue

                # Flatten the list of clauses into a single list of glosses
                flattened_gloss_list = [
                    gloss for clause in clause_glosses_list for gloss in clause
                ]

                # Skip if no glosses generated
                if not flattened_gloss_list:
                    skipped_count += 1
                    continue

                # Format glosses as space-separated string
                gloss_sequence = " ".join(flattened_gloss_list)
                text_sequence = original_text

                # Quality validation
                gloss_words = gloss_sequence.split()
                text_words = text_sequence.split()

                # Skip if gloss is suspiciously long
                if len(gloss_words) > len(text_words) * 3:
                    quality_filtered += 1
                    continue

                # Skip if gloss is too short
                if len(gloss_words) < 2:
                    quality_filtered += 1
                    continue

                # Skip if gloss is too long
                if len(gloss_words) > MAX_SENTENCE_WORDS:
                    quality_filtered += 1
                    continue

                # Skip if too much repetition
                if len(gloss_words) > 5:
                    unique_ratio = len(set(gloss_words)) / len(gloss_words)
                    if unique_ratio < 0.4:  # More than 60% duplicates
                        quality_filtered += 1
                        continue

                # Write valid data
                csv_writer.writerow([gloss_sequence, text_sequence])
                valid_count += 1

            except Exception as e:
                if conversion_errors < 10:  # Only print first few errors
                    print(f"Error processing sentence {i + 1}: {e}")
                conversion_errors += 1
                continue

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Valid samples generated: {valid_count:,}")
    print(f"Quality filtered: {quality_filtered:,}")
    print(f"Conversion errors: {conversion_errors:,}")
    print(f"Skipped (empty): {skipped_count:,}")
    print(f"Total processed: {len(processed_sentences):,}")
    print(f"Success rate: {valid_count / len(processed_sentences) * 100:.1f}%")
    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
