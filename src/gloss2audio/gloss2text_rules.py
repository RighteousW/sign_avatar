"""
Namibian Sign Language (NSL) Gloss to English Text Converter
Converts NSL gloss notation back to readable English text using spaCy for proper grammar.
"""

import spacy
from typing import List
from ..audio2gloss.word_mapping import _create_word_mapping
from ..audio2gloss.vocabulary import _get_default_vocabulary


class GlossToTextConverter:
    """
    Converts Namibian Sign Language (NSL) glosses back to English text.
    Uses spaCy to properly reconstruct grammar and handle SOV to SVO reordering.
    """

    def __init__(self, debug=False):
        self.nlp = None
        self.debug = debug

        # Reuse existing vocabulary and create reverse mapping
        valid_glosses = _get_default_vocabulary()
        word_to_gloss = _create_word_mapping(valid_glosses)

        # Create reverse mapping: gloss -> base English word
        self.gloss_to_word = {}

        # Map glosses to their base form for lemmatization
        gloss_base_forms = {
            "ME": "I",
            "ABDOMEN": "stomach",
            "BECOME_DEAF": "become deaf",
            "PARENTS": "parent",
            "RIDE_BICYCLE": "ride bicycle",
            "THANK_YOU": "thank you",
            "I_DISLIKE_YOU": "dislike you",
            "I_DONT_KNOW": "not know",
            "CHILD": "child",
            "CHILDREN": "child",
            "FRIEND": "friend",
            "BOOK": "book",
            "DOCTOR": "doctor",
            "WOMAN": "woman",
            "MAN": "man",
            "HELP": "help",
            "THROW": "throw",
            "MEET": "meet",
            "EAT": "eat",
            "GO": "go",
            "VOMIT": "vomit",
            "HOSPITAL": "hospital",
            "FOOD": "food",
            "SICK": "sick",
        }

        self.gloss_to_word.update(gloss_base_forms)

        # Add reverse mappings from word_to_gloss
        for word, gloss in word_to_gloss.items():
            if gloss not in self.gloss_to_word:
                self.gloss_to_word[gloss] = word

        # Ensure all valid glosses have a mapping
        for gloss in valid_glosses:
            if gloss not in self.gloss_to_word:
                self.gloss_to_word[gloss] = gloss.lower()

        # Define gloss categories for grammar reconstruction
        self.time_glosses = {
            "YESTERDAY",
            "TODAY",
            "TOMORROW",
            "BEFORE",
            "AFTER",
            "ALWAYS",
            "AGAIN",
            "SOON",
            "NOW",
        }
        self.location_glosses = {"HOSPITAL", "SCHOOL", "HOME", "PARK", "STORE"}

    def load_model(self):
        """Load the spaCy English language model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            return True
        except OSError:
            print(
                "Error: spaCy model not found. Install with: python -m spacy download en_core_web_sm"
            )
            return False

    def glosses_to_text(self, glosses: List[str]) -> str:
        """
        Convert a list of NSL glosses to grammatically correct English text.
        Uses spaCy to understand structure and properly inflect verbs.
        """
        if not glosses:
            return ""

        if not self.nlp:
            raise RuntimeError("spaCy model not loaded. Call load_model() first.")

        # Parse the gloss sequence
        components = self._parse_gloss_structure(glosses)

        if self.debug:
            print("\n" + "=" * 70)
            print("GLOSS TO TEXT CONVERSION")
            print("=" * 70)
            print(f"Input glosses: {' '.join(glosses)}")
            print(f"Parsed components: {components}")

        # Reconstruct English sentence with proper grammar
        text = self._reconstruct_sentence(components)

        if self.debug:
            print(f"Output text: {text}")
            print("=" * 70 + "\n")

        return text

    def _parse_gloss_structure(self, glosses: List[str]) -> dict:
        """
        Parse NSL gloss sequence into semantic components.
        NSL structure: TIME - LOCATION - SUBJECT - OBJECT - ADJECTIVE - VERB
        """
        components = {
            "time": [],
            "location": [],
            "subject": None,
            "objects": [],
            "adjectives": [],
            "verb": None,
            "other": [],
        }

        # Convert glosses to base words
        words = []
        for gloss in glosses:
            gloss_upper = gloss.upper()

            # Handle numbers
            if gloss_upper.isdigit():
                words.append((gloss_upper, gloss_upper, "NUM"))
                continue

            base_word = self.gloss_to_word.get(gloss_upper, gloss.lower())
            words.append((gloss_upper, base_word, None))

        # Classify components by position and type
        subject_found = False
        verb_found = False

        for i, (gloss, word, pos_hint) in enumerate(words):
            # Time expressions (usually first)
            if gloss in self.time_glosses:
                components["time"].append(word)
                continue

            # Location expressions
            if gloss in self.location_glosses and not subject_found:
                components["location"].append(word)
                continue

            # Analyze with spaCy
            doc = self.nlp(word)
            token = doc[0] if len(doc) > 0 else None

            if not token:
                components["other"].append(word)
                continue

            # Subject (first noun/pronoun before verb)
            if not subject_found and token.pos_ in ["NOUN", "PRON", "PROPN"]:
                components["subject"] = word
                subject_found = True
                continue

            # Object (noun after subject, before verb)
            if subject_found and not verb_found and token.pos_ in ["NOUN", "PROPN"]:
                components["objects"].append(word)
                continue

            # Adjectives
            if token.pos_ == "ADJ":
                components["adjectives"].append(word)
                continue

            # Verb (usually last in SOV structure)
            if token.pos_ in ["VERB", "AUX"] and token.lemma_ not in [
                "be",
                "have",
                "do",
            ]:
                components["verb"] = word
                verb_found = True
                continue

            components["other"].append(word)

        return components

    def _reconstruct_sentence(self, components: dict) -> str:
        """
        Reconstruct grammatically correct English sentence from components.
        """
        parts = []

        # Get subject
        subject = components["subject"]
        if not subject:
            subject = "it"

        # Get verb and conjugate it properly
        verb = components["verb"]
        if verb:
            # Use spaCy to get proper verb form
            conjugated_verb = self._conjugate_verb(verb, subject, components["time"])
            parts.append(subject)
            parts.append(conjugated_verb)
        else:
            parts.append(subject)

        # Add location with preposition if exists
        if components["location"]:
            parts.append("to the" if verb and verb in ["go", "went"] else "at the")
            parts.extend(components["location"])

        # Add objects with articles
        if components["objects"]:
            for obj in components["objects"]:
                # Add article if it's a singular countable noun
                doc = self.nlp(obj)
                if doc and doc[0].pos_ == "NOUN":
                    parts.append("the")
                parts.append(obj)

        # Add adjectives
        parts.extend(components["adjectives"])

        # Add other components
        parts.extend(components["other"])

        # Add time expressions at the end
        parts.extend(components["time"])

        # Join and capitalize
        text = " ".join(parts)
        if text:
            text = text[0].upper() + text[1:] + "."

        return text

    def _conjugate_verb(self, verb: str, subject: str, time_context: List[str]) -> str:
        """
        Conjugate verb based on subject and time context.
        """
        # Check if past time indicator
        past_indicators = {"yesterday", "before", "ago"}
        is_past = any(t in past_indicators for t in time_context)

        # Get lemma
        doc = self.nlp(verb)
        if not doc:
            return verb

        lemma = doc[0].lemma_

        # Simple conjugation rules
        if is_past:
            # Use past tense
            if lemma == "go":
                return "went"
            elif lemma == "eat":
                return "ate"
            elif lemma == "meet":
                return "met"
            elif lemma == "throw":
                return "threw"
            elif lemma == "vomit":
                return "vomited"
            elif lemma == "help":
                return "helped"
            else:
                # Regular past tense
                return lemma + "ed" if not lemma.endswith("e") else lemma + "d"
        else:
            # Present tense
            if subject.lower() in ["i", "you", "we", "they"]:
                return lemma
            else:
                # Third person singular
                if lemma.endswith(("s", "sh", "ch", "x", "z")):
                    return lemma + "es"
                elif lemma.endswith("y") and len(lemma) > 1:
                    return lemma[:-1] + "ies"
                else:
                    return lemma + "s"

    def clauses_to_text(self, clause_glosses: List[List[str]]) -> str:
        """Convert multiple clauses to full text"""
        sentences = []
        for glosses in clause_glosses:
            text = self.glosses_to_text(glosses)
            if text:
                sentences.append(text)

        return " ".join(sentences)


# Example usage
if __name__ == "__main__":
    converter = GlossToTextConverter(debug=True)

    if converter.load_model():
        # Test cases
        test_cases = [
            ["ME", "HOSPITAL", "YESTERDAY", "GO"],
            ["CHILD", "FOOD", "EAT"],
            ["ME", "FRIEND", "MEET"],
            ["ME", "BOOK", "THROW"],
            ["DOCTOR", "ME", "HELP"],
            ["SICK", "WOMAN", "VOMIT"],
        ]

        print("=== NSL Gloss to Text Converter ===\n")

        for glosses in test_cases:
            text = converter.glosses_to_text(glosses)
            print(f"Glosses: {' '.join(glosses)}")
            print(f"Text: {text}\n")
