import spacy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np


class Gloss2Voice:
    """
    Converts sign language glosses to natural English text.
    Reverses the voice-to-gloss transformations.
    """

    def __init__(self):
        """Initialize the converter with spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print(
                "Warning: spaCy model not loaded. Install with: python -m spacy download en_core_web_sm"
            )
            self.nlp = None

    def convert(self, glosses_text):
        """
        Convert glosses string to natural English.

        Args:
            glosses_text: String of space-separated glosses (e.g., "STORE I BOOK BUY NOT")

        Returns:
            Natural English sentence as string
        """
        glosses = glosses_text.upper().split()
        return self.glosses_to_natural_english(glosses)

    def glosses_to_natural_english(self, glosses):
        """
        Reverse the voice2gloss transformations:
        Original order: Location → Adverbs → Subject → Object → Verb → Negation
        Target order: Subject → Adverbs → Verb → Object → Location (with negation)
        """
        if not glosses:
            return ""

        # Step 1: Extract negation from end
        negation = []
        negation_words = ["NOT", "NO", "NEVER", "NONE"]
        while glosses and glosses[-1] in negation_words:
            negation.insert(0, glosses.pop())

        # Step 2: Extract location from start
        locations = []
        location_indicators = [
            "STORE",
            "HOME",
            "SCHOOL",
            "PARK",
            "WORK",
            "HOSPITAL",
            "LIBRARY",
            "MARKET",
            "CHURCH",
            "RESTAURANT",
            "OFFICE",
            "HERE",
            "THERE",
            "WHERE",
        ]
        while glosses and glosses[0] in location_indicators:
            locations.append(glosses.pop(0))

        # Step 3: Extract adverbs from start
        adverbs = []
        adverb_words = [
            "QUICKLY",
            "SLOWLY",
            "CAREFULLY",
            "FAST",
            "WELL",
            "BADLY",
            "ALWAYS",
            "OFTEN",
            "SOMETIMES",
            "RARELY",
            "USUALLY",
            "REALLY",
            "VERY",
            "TOO",
            "QUITE",
            "EXTREMELY",
        ]
        while glosses and glosses[0] in adverb_words:
            adverbs.append(glosses.pop(0))

        # Step 4: Parse SOV (Subject-Object-Verb)
        subject = None
        verb = None
        objects = []

        if len(glosses) >= 1:
            subject = glosses[0]
            if len(glosses) >= 2:
                verb = glosses[-1]
                if len(glosses) > 2:
                    objects = glosses[1:-1]

        # Step 5: Reconstruct as SVO with function words
        sentence_parts = []

        # Subject with article
        if subject:
            subject_lower = subject.lower()
            if subject_lower in ["i", "you", "he", "she", "it", "we", "they"]:
                sentence_parts.append(subject_lower)
            else:
                sentence_parts.append(self._add_article(subject_lower))

        # Adverbs
        if adverbs:
            sentence_parts.extend([adv.lower() for adv in adverbs])

        # Verb with negation
        if negation and verb:
            if subject and subject.lower() in ["he", "she", "it"]:
                sentence_parts.append("does not")
            else:
                sentence_parts.append("do not")
            sentence_parts.append(verb.lower())
        elif verb:
            verb_lower = verb.lower()
            if subject and subject.lower() in ["he", "she", "it"]:
                verb_lower = self._conjugate_third_person(verb_lower)
            sentence_parts.append(verb_lower)

        # Objects with articles
        for obj in objects:
            obj_lower = obj.lower()
            if obj_lower not in ["i", "you", "he", "she", "it", "we", "they"]:
                sentence_parts.append(self._add_article(obj_lower))
            else:
                sentence_parts.append(obj_lower)

        # Locations with prepositions
        if locations:
            for loc in locations:
                loc_lower = loc.lower()
                if loc_lower in ["here", "there"]:
                    sentence_parts.append(loc_lower)
                else:
                    prep = self._get_location_preposition(loc_lower)
                    sentence_parts.append(f"{prep} the {loc_lower}")

        # Capitalize and add period
        if sentence_parts:
            result = " ".join(sentence_parts)
            return result[0].upper() + result[1:] + "."
        return ""

    def _add_article(self, word):
        """Add appropriate article"""
        vowels = "aeiou"
        if word[0] in vowels:
            return f"an {word}"
        return f"a {word}"

    def _conjugate_third_person(self, verb):
        """Simple third person singular conjugation"""
        irregular = {
            "go": "goes",
            "do": "does",
            "have": "has",
            "be": "is",
            "say": "says",
        }
        if verb in irregular:
            return irregular[verb]
        elif verb.endswith(("s", "ss", "sh", "ch", "x", "z", "o")):
            return verb + "es"
        elif verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
            return verb[:-1] + "ies"
        else:
            return verb + "s"

    def _get_location_preposition(self, location):
        """Get appropriate preposition for location"""
        at_locations = [
            "home",
            "school",
            "work",
            "store",
            "hospital",
            "library",
            "church",
            "office",
            "park",
        ]
        return "at" if location in at_locations else "to"

    def evaluate(self, gloss_sequences, reference_sequences, verbose=False):
        """
        Evaluate the converter using BLEU scores.

        Args:
            gloss_sequences: List of gloss strings or list of gloss word lists
            reference_sequences: List of reference English strings or word lists
            verbose: If True, print example translations

        Returns:
            Dictionary with BLEU scores and examples
        """
        predictions = []
        references = []

        for i, (gloss, ref) in enumerate(zip(gloss_sequences, reference_sequences)):
            # Convert gloss to prediction
            if isinstance(gloss, str):
                pred = self.convert(gloss)
            else:
                pred = self.glosses_to_natural_english(gloss)

            # Convert to word lists for BLEU
            pred_words = pred.lower().replace(".", "").split()

            if isinstance(ref, str):
                ref_words = ref.lower().replace(".", "").split()
            else:
                ref_words = [w.lower() for w in ref]

            if pred_words and ref_words:
                predictions.append(pred_words)
                references.append(ref_words)

        # Calculate BLEU scores
        bleu_scores = self._calculate_bleu(predictions, references)

        # Get examples
        examples = []
        num_examples = min(5, len(predictions))
        for i in range(num_examples):
            gloss_input = (
                gloss_sequences[i]
                if isinstance(gloss_sequences[i], str)
                else " ".join(gloss_sequences[i])
            )
            examples.append(
                {
                    "gloss": gloss_input,
                    "prediction": " ".join(predictions[i]),
                    "reference": " ".join(references[i]),
                }
            )

        results = {
            "bleu_scores": bleu_scores,
            "examples": examples,
            "num_samples": len(predictions),
        }

        if verbose:
            print(f"Evaluated on {results['num_samples']} samples")
            print("\nBLEU Scores:")
            for metric, score in bleu_scores.items():
                print(f"  {metric}: {score:.4f} ({score*100:.2f}%)")

            print("\nExample Translations:")
            for i, ex in enumerate(examples, 1):
                print(f"\n{i}. Gloss: {ex['gloss']}")
                print(f"   Prediction: {ex['prediction']}")
                print(f"   Reference:  {ex['reference']}")

        return results

    def _calculate_bleu(self, predictions, references, max_n=4):
        """Calculate BLEU-1 through BLEU-4 scores"""
        smoothing = SmoothingFunction().method1
        bleu_scores = {f"BLEU-{i}": [] for i in range(1, max_n + 1)}

        for pred, ref in zip(predictions, references):
            for n in range(1, max_n + 1):
                weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
                score = sentence_bleu(
                    [ref], pred, weights=weights, smoothing_function=smoothing
                )
                bleu_scores[f"BLEU-{n}"].append(score)

        # Calculate means
        return {k: np.mean(v) for k, v in bleu_scores.items()}

    def evaluate_on_aslg(
        self,
        csv_path="data/dataset/synthetic/synthetic_ASLG-PC12.csv",
        max_samples=1000,
        verbose=True,
    ):
        """
        Evaluate on ASLG-PC12 dataset

        Args:
            csv_path: Path to ASLG-PC12 CSV file
            max_samples: Maximum number of samples to evaluate (None for all)
            verbose: Print results

        Returns:
            Dictionary with BLEU scores and examples
        """
        import csv

        gloss_sequences = []
        text_sequences = []

        # Load data
        print(f"Loading ASLG-PC12 dataset from {csv_path}...")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break

                gloss_text = row.get("gloss", row.get("Gloss", row.get("sign", "")))
                text_text = row.get("text", row.get("Text", row.get("english", "")))

                if gloss_text and text_text:
                    gloss = gloss_text.strip().split()
                    text = text_text.strip().split()

                    if gloss and text:
                        gloss_sequences.append(gloss)
                        text_sequences.append(text)

        print(f"Loaded {len(gloss_sequences)} valid samples")

        # Evaluate
        return self.evaluate(gloss_sequences, text_sequences, verbose=verbose)


# Update the example usage
if __name__ == "__main__":
    converter = Gloss2Voice()

    # Evaluate on ASLG-PC12 dataset
    print("=" * 60)
    print("EVALUATING RULE-BASED GLOSS2VOICE ON ASLG-PC12")
    print("=" * 60)

    results = converter.evaluate_on_aslg()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
