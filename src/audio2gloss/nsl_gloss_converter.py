import spacy


class NSLGlossConverter:

    def __init__(self):
        self.nlp = None

    def load_model(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            return True
        except OSError:
            print("Error: Install with: python3 -m spacy download en_core_web_sm")
            return False

    def text_to_gloss(self, text: str) -> dict:
        """Convert text to NSL glosses - naive approach"""
        doc = self.nlp(text)

        glosses = []

        for token in doc:
            # Skip auxiliaries (be, have, do)
            if token.lemma_ in ["be", "have", "do"]:
                continue

            # Skip articles
            if token.pos_ == "DET":
                continue

            # Handle negation
            if token.dep_ == "neg":
                glosses.append("NOT")
                continue

            # Handle conjunctions as markers
            if token.dep_ == "cc":
                continue  # Skip "and", "or" - handle with relations

            # Add content words
            if token.pos_ in [
                "NOUN",
                "VERB",
                "ADJ",
                "ADV",
                "PRON",
                "NUM",
                "PROPN",
                "PUNCT",
            ]:
                # Use lemma for verbs, text for others
                if token.pos_ == "VERB":
                    glosses.append(token.lemma_.upper())
                else:
                    glosses.append(token.text.upper())

        # Simple output
        return " ".join(glosses)


# Test
if __name__ == "__main__":
    converter = NSLGlossConverter()

    if converter.load_model():
        tests = [
            "I produce sputum sometimes and sometimes I don't.",
            "If it rains, we will stay home.",
            "I have one apple, two oranges and three bananas.",
            "Two plus two equals four.",
            "She quickly ran to the store, but it was closed.",
            "He is not going to the party because he is sick.",
            "Although it was raining, they decided to go hiking.",
            "Can you believe it's already September?",
            "The cat sat on the mat.",
            "They've been waiting for hours!",
            "Sapphire is the most precious and valuable blue gemstone.",
            "Alarm interventions for nocturnal enuresis in children.",
            "Did this chiropractor use current and safe equipment?"
        ]

        for text in tests:
            result = converter.text_to_gloss(text)
            print(f"Original: {text}")
            print(f"Gloss: {result}")
            print()
