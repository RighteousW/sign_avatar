"""
Namibian Sign Language (NSL) Gloss Converter
Converts English audio/text to NSL gloss notation following NSL grammar rules.
The converter follows NSL structure: TIME - LOCATION - SUBJECT - OBJECT -
ADJECTIVE - VERB - NEGATION - QUESTION
Following:
1. @inproceedings{moryossef2023baseline,
 title={An Open-Source Gloss-Based Baseline for Spoken to Signed Language Translation},
 author={Moryossef, Amit and M{\"u}ller, Mathias and G{\"o}hring, Anne and Jiang, Zifan and Goldberg, Yoav and Ebling, Sarah},
 booktitle={2nd International Workshop on Automatic Translation for Signed and Spoken Languages (AT4SSL)},
 year={2023},
 month={June},
 url={https://github.com/ZurichNLP/spoken-to-signed-translation},
 note={Available at: \\url{https://arxiv.org/abs/2305.17714}}
}
2. Morgan, Ruth; Liddell, Scott; Haikali, Marius M.N.; Ashipala, Sackeus P.; Daniel, Polo; Haiduwah, Hilifilua
E.T.; Hashiyana, Rauna Ndeshihafela; Israel, Nangolo Jeremia; Linus, Festus Tshikuku; Niilenge, Henock
Hango; and Setzer, Paul, "Namibian Sign Language to English and Oshiwambo" (1991). Namibian Sign
Language to English and Oshiwambo. 1 Front matter xxi-xxviii.
"""

import speech_recognition as sr
import spacy
import io
from typing import Tuple, List


class AudioToGlossConverter:
    """
    Converts audio files to Namibian Sign Language (NSL) glosses.

    FIXED VERSION: Properly handles hyphenated compounds and prepositional phrases.
    """

    def __init__(self, debug=False):
        self.recognizer = sr.Recognizer()
        self.nlp = None
        self.target_language = "NSL"
        self.debug = debug

    def load_model(self):
        """Load the spaCy English language model for NLP processing"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            return True
        except OSError:
            print(
                "Error: spaCy model not found. Install with: python -m spacy download en_core_web_sm"
            )
            return False

    def audio_data_to_text(self, audio_data: sr.AudioData) -> str:
        """Convert an AudioData object to text using Google's speech recognition"""
        text = self.recognizer.recognize_google(audio_data)
        return text

    def audio_file_to_text(self, audio_file_path: str) -> str:
        """Convert an audio file (WAV format) to text"""
        with sr.AudioFile(audio_file_path) as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = self.recognizer.record(source)
        text = self.recognizer.recognize_google(audio)
        return text

    def numpy_to_audio_data(self, audio_array, sample_rate: int) -> sr.AudioData:
        """Convert a numpy array to an AudioData object"""
        import numpy as np
        import wave

        if len(audio_array.shape) > 1:
            audio_array = audio_array.flatten()

        audio_clipped = np.clip(audio_array, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)

        byte_io = io.BytesIO()
        with wave.open(byte_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        byte_io.seek(0)
        wav_data = byte_io.read()
        audio_data = sr.AudioData(wav_data, sample_rate, 2)
        return audio_data

    def text_to_glosses(self, text: str) -> List[List[str]]:
        """Convert text to Namibian Sign Language glosses"""
        if not self.nlp:
            raise RuntimeError("spaCy model not loaded. Call load_model() first.")

        doc = self.nlp(text)
        all_clause_glosses = []

        for sent in doc.sents:
            clause_glosses = self._sentence_to_glosses(sent)
            if clause_glosses:
                all_clause_glosses.append(clause_glosses)

        return all_clause_glosses

    def audio_file_to_glosses(
        self, audio_file_path: str
    ) -> Tuple[str, List[List[str]]]:
        """Convert audio file directly to NSL glosses"""
        text = self.audio_file_to_text(audio_file_path)
        clause_glosses = self.text_to_glosses(text)
        return text, clause_glosses

    def audio_data_to_glosses(
        self, audio_data: sr.AudioData
    ) -> Tuple[str, List[List[str]]]:
        """Convert AudioData directly to NSL glosses"""
        text = self.audio_data_to_text(audio_data)
        clause_glosses = self.text_to_glosses(text)
        return text, clause_glosses

    def numpy_to_glosses(
        self, audio_array, sample_rate: int
    ) -> Tuple[str, List[List[str]]]:
        """Convert numpy array directly to NSL glosses"""
        audio_data = self.numpy_to_audio_data(audio_array, sample_rate)
        return self.audio_data_to_glosses(audio_data)

    def _reconstruct_compounds(self, tokens):
        """
        Reconstruct hyphenated compounds that spaCy splits.
        Returns a dict mapping token indices to their compound forms.

        Example: "cancer - related" -> {idx_of_related: "CANCER-RELATED"}
        """
        compounds = {}
        i = 0

        while i < len(tokens):
            # Look for pattern: word + hyphen + word
            if i + 2 < len(tokens):
                if tokens[i + 1].text == "-" or tokens[i + 1].tag_ == "HYPH":
                    # Found hyphen, combine tokens using ORIGINAL TEXT (not lemma)
                    # This preserves "cancer-related" instead of "cancer-relate"
                    parts = [tokens[i].text.upper()]
                    j = i + 2

                    # Add the part after hyphen
                    parts.append(tokens[j].text.upper())

                    # Check for multiple hyphens (e.g., "well-known-fact")
                    while j + 2 < len(tokens) and tokens[j + 1].text == "-":
                        parts.append(tokens[j + 2].text.upper())
                        j += 2

                    # The last token in the compound is the one we'll keep
                    compound = "-".join(parts)
                    compounds[j] = compound

                    # Mark the earlier tokens as used
                    for k in range(i, j):
                        compounds[k] = None  # Mark as consumed

                    i = j + 1
                    continue

            i += 1

        return compounds

    def _get_gloss_token(self, token, compounds_map) -> str:
        """
        Convert a single token to its gloss representation.
        Checks if it's part of a compound first.
        """
        # Check if this token is part of a compound
        if token.i in compounds_map:
            compound = compounds_map[token.i]
            if compound is None:  # This token was consumed by a compound
                return None
            else:  # This is the main token of the compound
                return compound

        # Possessive pronouns stay as-is
        if token.tag_ in ["PRP$", "WP$"]:
            return token.text.upper()

        # Remove possessive 's
        if "'s" in token.text:
            return token.text.replace("'s", "").upper()

        # Preserve plurals for nouns
        if token.pos_ == "NOUN" and token.tag_ in ["NNS", "NNPS"]:
            return token.text.upper()

        # Default: use lemma
        return token.lemma_.upper()

    def _is_content_word(self, token) -> bool:
        """Check if token should be included in glosses"""
        # Keep content words
        if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV", "NUM", "PRON", "PROPN"]:
            # Skip auxiliary verbs (be, have, do)
            if token.pos_ == "VERB" and token.lemma_ in ["be", "have", "do"]:
                return False
            # Skip auxiliary "be" tagged as AUX
            if token.pos_ == "AUX" and token.lemma_ == "be":
                return False
            # Skip expletive "it"
            if token.text.lower() == "it" and token.dep_ == "expl":
                return False
            return True

        # Keep modals
        if token.tag_ == "MD":
            return True

        # Keep important prepositions (only when they're standalone, not part of noun phrases)
        if token.pos_ == "ADP":
            important_preps = {
                "with",
                "without",
                "for",
                "about",
                "from",
                "to",
                "in",
                "on",
                "at",
                "by",
                "after",
                "before",
                "during",
                "until",
                "since",
                "through",
                "near",
                "under",
                "over",
            }
            return token.text.lower() in important_preps

        return False

    def _get_noun_phrase_tokens(self, head_token, compounds_map):
        """
        Get all tokens that are part of a noun phrase.
        Returns list of tokens in correct order, handling compounds.
        """
        phrase_tokens = []

        # Collect modifiers
        for child in head_token.children:
            # Possessives (MY, HIS, THEIR)
            if child.dep_ == "poss" and child.i < head_token.i:
                phrase_tokens.append(child)

            # Compounds (like "Sri" in "Sri Lanka")
            elif child.dep_ == "compound" and child.i < head_token.i:
                phrase_tokens.append(child)

            # Adjectives (including compound adjectives like "cancer-related")
            elif child.dep_ == "amod":
                # Check if this adjective has modifiers (like "cancer" modifying "related")
                for grandchild in child.children:
                    if (
                        grandchild.dep_ in ["npadvmod", "advmod"]
                        and grandchild.i < child.i
                    ):
                        # This will be part of a compound, let _reconstruct_compounds handle it
                        pass
                phrase_tokens.append(child)

            # Numbers
            elif child.dep_ == "nummod" and child.i < head_token.i:
                phrase_tokens.append(child)

        # Add the head noun itself
        phrase_tokens.append(head_token)

        # Sort by position
        phrase_tokens.sort(key=lambda t: t.i)

        return phrase_tokens

    def _sentence_to_glosses(self, sent) -> List[str]:
        """
        Convert a sentence to NSL glosses with proper SOV ordering.

        NSL structure: TIME/TOPIC - SUBJECT - OBJECT - MODIFIERS - VERB - NEGATION
        """
        glosses = []
        tokens = list(sent)
        added = set()

        # Step 1: Reconstruct compounds
        compounds_map = self._reconstruct_compounds(tokens)

        if self.debug:
            print(f"\n=== Processing: {sent.text} ===")
            print(
                "Compounds found:",
                {k: v for k, v in compounds_map.items() if v is not None},
            )

        # Step 2: Time expressions and discourse markers (go first)
        time_markers = [
            "now",
            "today",
            "yesterday",
            "tomorrow",
            "soon",
            "later",
            "however",
            "therefore",
            "thus",
            "meanwhile",
            "then",
        ]

        for token in tokens:
            if token.text.lower() in time_markers and self._is_content_word(token):
                if token not in added:
                    gloss = self._get_gloss_token(token, compounds_map)
                    if gloss:
                        glosses.append(gloss)
                        added.add(token)

        # Step 3: Identify sentence components
        subjects = []
        direct_objects = []  # Only true direct objects, not prepositional objects
        verbs = []
        modals = []
        adverbs = []

        for token in tokens:
            if token in added:
                continue

            # Subjects
            if token.dep_ in ["nsubj", "nsubjpass"]:
                subjects.append(token)

            # Direct objects only (NOT prepositional objects)
            elif token.dep_ in ["dobj", "attr"]:
                direct_objects.append(token)

            # Main verbs (not auxiliaries)
            elif token.pos_ == "VERB" and token.dep_ not in ["aux", "auxpass"]:
                if token.lemma_ not in ["be", "have", "do"]:
                    verbs.append(token)

            # Modals
            elif token.tag_ == "MD":
                modals.append(token)

            # Adverbs that modify verbs
            elif token.pos_ == "ADV" and token.dep_ == "advmod":
                if token.head.pos_ in ["VERB", "AUX"]:
                    adverbs.append(token)

        # Step 4: Build glosses in NSL order

        # SUBJECTS (with their modifiers)
        for subj in subjects:
            phrase_tokens = self._get_noun_phrase_tokens(subj, compounds_map)

            for token in phrase_tokens:
                if token not in added:
                    gloss = self._get_gloss_token(token, compounds_map)
                    if gloss:
                        glosses.append(gloss)
                        added.add(token)

        # DIRECT OBJECTS (with their modifiers)
        for obj in direct_objects:
            phrase_tokens = self._get_noun_phrase_tokens(obj, compounds_map)

            for token in phrase_tokens:
                if token not in added:
                    gloss = self._get_gloss_token(token, compounds_map)
                    if gloss:
                        glosses.append(gloss)
                        added.add(token)

        # PREPOSITIONAL PHRASES (keep preposition + object together)
        # These provide context (location, manner, etc.)
        for token in tokens:
            if token.pos_ == "ADP" and self._is_content_word(token):
                if token not in added:
                    # Add the preposition
                    gloss = self._get_gloss_token(token, compounds_map)
                    if gloss:
                        glosses.append(gloss)
                        added.add(token)

                    # Add the object of the preposition
                    for child in token.children:
                        if child.dep_ == "pobj":
                            obj_phrase = self._get_noun_phrase_tokens(
                                child, compounds_map
                            )
                            for obj_token in obj_phrase:
                                if obj_token not in added:
                                    obj_gloss = self._get_gloss_token(
                                        obj_token, compounds_map
                                    )
                                    if obj_gloss:
                                        glosses.append(obj_gloss)
                                        added.add(obj_token)

        # REMAINING CONTENT WORDS (catch anything we missed)
        # This ensures we don't lose important nouns, adjectives, etc.
        for token in tokens:
            if self._is_content_word(token) and token not in added:
                gloss = self._get_gloss_token(token, compounds_map)
                if gloss:
                    glosses.append(gloss)
                    added.add(token)

        # MODALS (before verbs in NSL)
        for modal in modals:
            if modal not in added:
                gloss = self._get_gloss_token(modal, compounds_map)
                if gloss:
                    glosses.append(gloss)
                    added.add(modal)

        # VERBS (main action)
        for verb in verbs:
            if verb not in added:
                gloss = self._get_gloss_token(verb, compounds_map)
                if gloss:
                    glosses.append(gloss)
                    added.add(verb)

        # VERB ADVERBS (like "sooner")
        for adv in adverbs:
            if adv not in added:
                gloss = self._get_gloss_token(adv, compounds_map)
                if gloss:
                    glosses.append(gloss)
                    added.add(adv)

        # NEGATION (at the end)
        for token in tokens:
            if token.dep_ == "neg" or (
                token.text.lower() in ["not", "never"] and token.pos_ == "PART"
            ):
                if "NOT" not in glosses and "NEVER" not in glosses:
                    if token.text.lower() == "never":
                        glosses.append("NEVER")
                    else:
                        glosses.append("NOT")
                break

        if self.debug:
            print(f"Output: {' '.join(glosses)}")

        return glosses


# Example usage and testing
if __name__ == "__main__":
    # Create converter with debug mode disabled by default
    converter = AudioToGlossConverter(debug=False)

    if converter.load_model():
        test_sentences = [
            # Basic sentences
            "Hospital after an accident",
            "I am going to the store tomorrow",
            "The dog is not eating his food",
            "She quickly ran to the park",
            # Questions
            "Where is the cat?",
            "What did you eat for breakfast?",
            "When will the meeting start?",
            # Negations
            "I don't like vegetables",
            "She will never forget this moment",
            "He hasn't finished his homework yet",
            # Conditionals
            "If it rains, we will stay home",
            "When she arrives, we can start dinner",
            # Multiple clauses
            "I went to the store and bought some milk",
            "Because it was cold, I wore my heavy jacket",
            # Possessives
            "My brother's car is very fast",
            "Their teacher gave them a difficult assignment",
            # Time expressions
            "Yesterday I visited my grandmother",
            "Next week we will travel to Paris",
            "Last night the storm damaged several houses",
            # Spatial concepts (important for NSL)
            "The children argue with the teacher",
            "People are talking to me from all sides",
            "She gave the book to him",
        ]

        print("=== Namibian Sign Language (NSL) Gloss Converter ===\n")

        for sentence in test_sentences:
            clause_glosses = converter.text_to_glosses(sentence)

            print(f"Input: {sentence}")
            print(f"Output: {len(clause_glosses)} clause(s)")

            for i, glosses in enumerate(clause_glosses, 1):
                print(f"  Clause {i}: {' '.join(glosses)}")

            print()
