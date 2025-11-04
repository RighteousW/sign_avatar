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
        """
        compounds = {}
        i = 0

        while i < len(tokens):
            if i + 2 < len(tokens):
                if tokens[i + 1].text == "-" or tokens[i + 1].tag_ == "HYPH":
                    parts = [tokens[i].text.upper()]
                    j = i + 2
                    parts.append(tokens[j].text.upper())

                    while j + 2 < len(tokens) and tokens[j + 1].text == "-":
                        parts.append(tokens[j + 2].text.upper())
                        j += 2

                    compound = "-".join(parts)
                    compounds[j] = compound

                    for k in range(i, j):
                        compounds[k] = None

                    i = j + 1
                    continue
            i += 1

        return compounds

    def _get_gloss_token(self, token, compounds_map) -> str:
        """Convert a single token to its gloss representation."""
        if token.i in compounds_map:
            compound = compounds_map[token.i]
            if compound is None:
                return None
            else:
                return compound

        if token.tag_ in ["PRP$", "WP$"]:
            return token.text.upper()

        if "'s" in token.text:
            return token.text.replace("'s", "").upper()

        if token.pos_ == "NOUN" and token.tag_ in ["NNS", "NNPS"]:
            return token.text.upper()

        return token.lemma_.upper()

    def _is_content_word(self, token) -> bool:
        """Check if token should be included in glosses"""
        if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV", "NUM", "PRON", "PROPN"]:
            if token.pos_ == "VERB" and token.lemma_ in ["be", "have", "do"]:
                return False
            if token.pos_ == "AUX" and token.lemma_ == "be":
                return False
            if token.text.lower() == "it" and token.dep_ == "expl":
                return False
            return True

        if token.tag_ == "MD":
            return True

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

        FIX: Use set to prevent duplicates.
        """
        phrase_token_set = set()  # Use set to prevent duplicates
        phrase_tokens = []

        # Collect modifiers
        for child in head_token.children:
            if child.dep_ == "poss" and child.i < head_token.i:
                if child not in phrase_token_set:
                    phrase_token_set.add(child)
                    phrase_tokens.append(child)

            elif child.dep_ == "compound" and child.i < head_token.i:
                if child not in phrase_token_set:
                    phrase_token_set.add(child)
                    phrase_tokens.append(child)

            elif child.dep_ == "amod":
                if child not in phrase_token_set:
                    phrase_token_set.add(child)
                    phrase_tokens.append(child)

            elif child.dep_ == "nummod" and child.i < head_token.i:
                if child not in phrase_token_set:
                    phrase_token_set.add(child)
                    phrase_tokens.append(child)

        # Add the head noun itself
        if head_token not in phrase_token_set:
            phrase_token_set.add(head_token)
            phrase_tokens.append(head_token)

        # Sort by position
        phrase_tokens.sort(key=lambda t: t.i)

        return phrase_tokens


    def _sentence_to_glosses(self, sent) -> List[str]:
        """
        Convert a sentence to NSL glosses with proper SOV ordering.
        Enhanced debug output showing all processing steps.
        """
        glosses = []
        tokens = list(sent)
        added_tokens = set()
        added_glosses = set()

        # Step 1: Reconstruct compounds
        compounds_map = self._reconstruct_compounds(tokens)

        if self.debug:
            print("\n" + "=" * 70)
            print("ENGLISH SENTENCE")
            print("=" * 70)
            print(f"Input: {sent.text}")
            print()

            print("=" * 70)
            print("SPACY NLP PROCESSING")
            print("=" * 70)
            print(f"{'Token':<15} {'POS':<10} {'Tag':<10} {'Dep':<15} {'Head':<15}")
            print("-" * 70)
            for token in tokens:
                print(
                    f"{token.text:<15} {token.pos_:<10} {token.tag_:<10} "
                    f"{token.dep_:<15} {token.head.text:<15}"
                )
            print()

            if any(v is not None for v in compounds_map.values()):
                print("Compounds Reconstructed:")
                for k, v in compounds_map.items():
                    if v is not None:
                        print(f"  Token {k}: {v}")
                print()

        # Helper function to add gloss safely
        def add_gloss_safe(token):
            """Add gloss only if not already added"""
            if token in added_tokens:
                return False

            gloss = self._get_gloss_token(token, compounds_map)
            if gloss and gloss not in added_glosses:
                glosses.append(gloss)
                added_tokens.add(token)
                added_glosses.add(gloss)
                return True

            added_tokens.add(token)
            return False

        # Step 2: Time expressions and discourse markers
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

        time_expressions = []
        for token in tokens:
            if token.text.lower() in time_markers and self._is_content_word(token):
                time_expressions.append(token.text)
                add_gloss_safe(token)

        # Step 3: Identify sentence components
        subjects = []
        direct_objects = []
        verbs = []
        modals = []
        adverbs = []
        prepositions = []

        for token in tokens:
            if token in added_tokens:
                continue

            if token.dep_ in ["nsubj", "nsubjpass"]:
                subjects.append(token)
            elif token.dep_ in ["dobj", "attr"]:
                direct_objects.append(token)
            elif token.pos_ == "VERB" and token.dep_ not in ["aux", "auxpass"]:
                if token.lemma_ not in ["be", "have", "do"]:
                    verbs.append(token)
            elif token.tag_ == "MD":
                modals.append(token)
            elif token.pos_ == "ADV" and token.dep_ == "advmod":
                if token.head.pos_ in ["VERB", "AUX"]:
                    adverbs.append(token)
            elif token.pos_ == "ADP" and self._is_content_word(token):
                prepositions.append(token)

        if self.debug:
            print("=" * 70)
            print("COMPONENT IDENTIFICATION")
            print("=" * 70)
            if time_expressions:
                print(f"Time/Discourse: {', '.join(time_expressions)}")
            if subjects:
                print(f"Subjects: {', '.join([s.text for s in subjects])}")
            if direct_objects:
                print(f"Objects: {', '.join([o.text for o in direct_objects])}")
            if verbs:
                print(f"Verbs: {', '.join([v.text for v in verbs])}")
            if modals:
                print(f"Modals: {', '.join([m.text for m in modals])}")
            if adverbs:
                print(f"Adverbs: {', '.join([a.text for a in adverbs])}")
            if prepositions:
                print(f"Prepositions: {', '.join([p.text for p in prepositions])}")

            # Check for negation
            negation_tokens = [
                t.text
                for t in tokens
                if t.dep_ == "neg"
                or (t.text.lower() in ["not", "never"] and t.pos_ == "PART")
            ]
            if negation_tokens:
                print(f"Negation: {', '.join(negation_tokens)}")
            print()

        # Step 4: Build glosses in NSL order
        if self.debug:
            print("=" * 70)
            print("SOV REORDERING + FUNCTION WORD REMOVAL")
            print("=" * 70)
            step_num = 1

        # SUBJECTS
        if subjects and self.debug:
            print(f"Step {step_num}: Adding SUBJECTS")
            step_num += 1
        for subj in subjects:
            phrase_tokens = self._get_noun_phrase_tokens(subj, compounds_map)
            if self.debug and phrase_tokens:
                phrase_words = [t.text for t in phrase_tokens]
                print(f"  Noun phrase: {' '.join(phrase_words)}")
            for token in phrase_tokens:
                add_gloss_safe(token)

        # DIRECT OBJECTS
        if direct_objects and self.debug:
            print(f"Step {step_num}: Adding DIRECT OBJECTS")
            step_num += 1
        for obj in direct_objects:
            phrase_tokens = self._get_noun_phrase_tokens(obj, compounds_map)
            if self.debug and phrase_tokens:
                phrase_words = [t.text for t in phrase_tokens]
                print(f"  Noun phrase: {' '.join(phrase_words)}")
            for token in phrase_tokens:
                add_gloss_safe(token)

        # PREPOSITIONAL PHRASES
        prep_added = False
        for token in tokens:
            if token.pos_ == "ADP" and self._is_content_word(token):
                if token not in added_tokens:
                    if not prep_added and self.debug:
                        print(f"Step {step_num}: Adding PREPOSITIONAL PHRASES")
                        step_num += 1
                        prep_added = True

                    if self.debug:
                        print(f"  Preposition: {token.text}")
                    add_gloss_safe(token)

                    # Add object of preposition
                    for child in token.children:
                        if child.dep_ == "pobj":
                            obj_phrase = self._get_noun_phrase_tokens(child, compounds_map)
                            if self.debug and obj_phrase:
                                phrase_words = [t.text for t in obj_phrase]
                                print(f"    Object: {' '.join(phrase_words)}")
                            for obj_token in obj_phrase:
                                add_gloss_safe(obj_token)

        # REMAINING CONTENT WORDS
        remaining = []
        for token in tokens:
            if self._is_content_word(token) and token not in added_tokens:
                remaining.append(token.text)
                add_gloss_safe(token)
        if remaining and self.debug:
            print(f"Step {step_num}: Adding OTHER CONTENT WORDS")
            print(f"  Words: {', '.join(remaining)}")
            step_num += 1

        # MODALS
        if modals and self.debug:
            print(f"Step {step_num}: Adding MODALS")
            print(f"  Words: {', '.join([m.text for m in modals])}")
            step_num += 1
        for modal in modals:
            add_gloss_safe(modal)

        # VERBS
        if verbs and self.debug:
            print(f"Step {step_num}: Adding VERBS")
            print(f"  Words: {', '.join([v.text for v in verbs])}")
            step_num += 1
        for verb in verbs:
            add_gloss_safe(verb)

        # VERB ADVERBS
        if adverbs and self.debug:
            print(f"Step {step_num}: Adding ADVERBS")
            print(f"  Words: {', '.join([a.text for a in adverbs])}")
            step_num += 1
        for adv in adverbs:
            add_gloss_safe(adv)

        # NEGATION
        negation_found = False
        for token in tokens:
            if not negation_found and (
                token.dep_ == "neg"
                or (token.text.lower() in ["not", "never"] and token.pos_ == "PART")
            ):
                if self.debug:
                    print(f"Step {step_num}: Adding NEGATION")
                    print(f"  Word: {token.text}")

                if token.text.lower() == "never":
                    if "NEVER" not in added_glosses:
                        glosses.append("NEVER")
                        added_glosses.add("NEVER")
                else:
                    if "NOT" not in added_glosses:
                        glosses.append("NOT")
                        added_glosses.add("NOT")
                negation_found = True
                break

        if self.debug:
            print()
            print("=" * 70)
            print("NSL GLOSS SEQUENCE")
            print("=" * 70)
            print(f"Output: {' '.join(glosses)}")
            print("=" * 70)
            print()

        return glosses


# Example usage and testing
if __name__ == "__main__":
    # Create converter with debug mode disabled by default
    converter = AudioToGlossConverter(debug=True)

    if converter.load_model():
        # test_sentences = [
        #     # Basic sentences
        #     "Hospital after an accident",
        #     "I am going to the store tomorrow",
        #     "The dog is not eating his food",
        #     "She quickly ran to the park",
        #     # Questions
        #     "Where is the cat?",
        #     "What did you eat for breakfast?",
        #     "When will the meeting start?",
        #     # Negations
        #     "I don't like vegetables",
        #     "She will never forget this moment",
        #     "He hasn't finished his homework yet",
        #     # Conditionals
        #     "If it rains, we will stay home",
        #     "When she arrives, we can start dinner",
        #     # Multiple clauses
        #     "I went to the store and bought some milk",
        #     "Because it was cold, I wore my heavy jacket",
        #     # Possessives
        #     "My brother's car is very fast",
        #     "Their teacher gave them a difficult assignment",
        #     # Time expressions
        #     "Yesterday I visited my grandmother",
        #     "Next week we will travel to Paris",
        #     "Last night the storm damaged several houses",
        #     # Spatial concepts (important for NSL)
        #     "The children argue with the teacher",
        #     "People are talking to me from all sides",
        #     "She gave the book to him",
        # ]
        test_sentences = ["I went to the hospital yesterday"]

        print("=== Namibian Sign Language (NSL) Gloss Converter ===\n")

        for sentence in test_sentences:
            clause_glosses = converter.text_to_glosses(sentence)

            print(f"Input: {sentence}")
            print(f"Output: {len(clause_glosses)} clause(s)")

            for i, glosses in enumerate(clause_glosses, 1):
                print(f"  Clause {i}: {' '.join(glosses)}")

            print()
