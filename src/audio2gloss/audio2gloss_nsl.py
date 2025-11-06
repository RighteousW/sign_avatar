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
from typing import Tuple, List, Optional, Set


class AudioToGlossConverter:
    """
    Converts audio files to Namibian Sign Language (NSL) glosses.
    CONSTRAINED to only use available gesture recognizer vocabulary.
    """

    def __init__(self, valid_glosses: Optional[List[str]] = None, debug=False):
        self.recognizer = sr.Recognizer()
        self.nlp = None
        self.target_language = "NSL"
        self.debug = debug

        # Define valid vocabulary from gesture recognizer
        if valid_glosses is None:
            self.valid_glosses = self._get_default_vocabulary()
        else:
            self.valid_glosses = set(g.upper() for g in valid_glosses)

        # Create mapping for word variations to valid glosses
        self.word_to_gloss_map = self._create_word_mapping()

        if self.debug:
            print(f"Loaded {len(self.valid_glosses)} valid glosses")

    def _get_default_vocabulary(self) -> Set[str]:
        """Default vocabulary from your gesture recognizer class names"""
        return {
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "ABDOMEN",
            "ACCIDENT",
            "AFTER",
            "AGAIN",
            "AGO",
            "ALCOHOL",
            "ALSO",
            "ALWAYS",
            "ANGRY",
            "ANIMAL",
            "ARRIVE",
            "BANANA",
            "BECOME_DEAF",
            "BEFORE",
            "BEGIN",
            "BIRD",
            "BLOOD",
            "BOOK",
            "BORING",
            "BORN",
            "BOSS",
            "BOX",
            "BOY",
            "BREAST_FEED",
            "BROKEN",
            "BUT",
            "CAR",
            "CHILD",
            "CHILDREN",
            "CHURCH",
            "CLEAN",
            "CLEVER",
            "COME",
            "DIFFICULT",
            "DOCTOR",
            "DRINK",
            "FEVERISH",
            "FRIEND",
            "GIRL",
            "GIVE",
            "GRANDFATHER",
            "GRANDMOTHER",
            "HAVE",
            "HEALTHY",
            "HIGH",
            "I_DISLIKE_YOU",
            "I_DONT_KNOW",
            "KIDNEY",
            "LOW",
            "MAN",
            "ME",
            "MEET",
            "MOTORBIKE",
            "OTHERS",
            "PARENTS",
            "PERSON_STRADDLES_BRANCH",
            "PERSON_SWINGS_FROM_BRANCH",
            "PROTECT",
            "REMEMBER",
            "RIDE_BICYCLE",
            "RUN",
            "SEE",
            "SIBLING",
            "SICK",
            "SPEAK",
            "STUPID",
            "THANK_YOU",
            "THROW",
            "TIME",
            "VIOLENCE",
            "VOMIT",
            "WHAT",
            "WITH",
            "WOMAN",
            "YOU_DONT_KNOW_ANYTHING",
        }

    def _create_word_mapping(self) -> dict:
        """Create mapping from English words/lemmas to valid glosses"""
        mapping = {}

        # Direct mappings
        word_mappings = {
            # Numbers
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            # Common words
            "i": "ME",
            "me": "ME",
            "my": "ME",
            "myself": "ME",
            "stomach": "ABDOMEN",
            "belly": "ABDOMEN",
            "tummy": "ABDOMEN",
            "crash": "ACCIDENT",
            "collision": "ACCIDENT",
            "later": "AFTER",
            "afterwards": "AFTER",
            "once more": "AGAIN",
            "repeat": "AGAIN",
            "liquor": "ALCOHOL",
            "beer": "ALCOHOL",
            "wine": "ALCOHOL",
            "too": "ALSO",
            "as well": "ALSO",
            "forever": "ALWAYS",
            "constantly": "ALWAYS",
            "mad": "ANGRY",
            "furious": "ANGRY",
            "upset": "ANGRY",
            "creature": "ANIMAL",
            "beast": "ANIMAL",
            "reach": "ARRIVE",
            "get": "ARRIVE",
            "come": "ARRIVE",
            "start": "BEGIN",
            "commence": "BEGIN",
            "deaf": "BECOME_DEAF",
            "deafened": "BECOME_DEAF",
            "earlier": "BEFORE",
            "prior": "BEFORE",
            "kid": "CHILD",
            "baby": "CHILD",
            "toddler": "CHILD",
            "kids": "CHILDREN",
            "babies": "CHILDREN",
            "vehicle": "CAR",
            "automobile": "CAR",
            "smart": "CLEVER",
            "intelligent": "CLEVER",
            "bright": "CLEVER",
            "hard": "DIFFICULT",
            "challenging": "DIFFICULT",
            "tough": "DIFFICULT",
            "physician": "DOCTOR",
            "doc": "DOCTOR",
            "medic": "DOCTOR",
            "beverage": "DRINK",
            "sip": "DRINK",
            "hot": "FEVERISH",
            "fever": "FEVERISH",
            "pal": "FRIEND",
            "buddy": "FRIEND",
            "mate": "FRIEND",
            "lady": "GIRL",
            "lass": "GIRL",
            "provide": "GIVE",
            "donate": "GIVE",
            "offer": "GIVE",
            "grandpa": "GRANDFATHER",
            "granddad": "GRANDFATHER",
            "grandma": "GRANDMOTHER",
            "granny": "GRANDMOTHER",
            "nana": "GRANDMOTHER",
            "possess": "HAVE",
            "own": "HAVE",
            "well": "HEALTHY",
            "fit": "HEALTHY",
            "fine": "HEALTHY",
            "tall": "HIGH",
            "elevated": "HIGH",
            "short": "LOW",
            "small": "LOW",
            "guy": "MAN",
            "male": "MAN",
            "gentleman": "MAN",
            "encounter": "MEET",
            "greet": "MEET",
            "motorcycle": "MOTORBIKE",
            "bike": "MOTORBIKE",
            "rest": "OTHERS",
            "remaining": "OTHERS",
            "mom": "PARENTS",
            "dad": "PARENTS",
            "mother": "PARENTS",
            "father": "PARENTS",
            "guard": "PROTECT",
            "defend": "PROTECT",
            "shield": "PROTECT",
            "recall": "REMEMBER",
            "recollect": "REMEMBER",
            "cycle": "RIDE_BICYCLE",
            "bike": "RIDE_BICYCLE",
            "pedal": "RIDE_BICYCLE",
            "jog": "RUN",
            "sprint": "RUN",
            "dash": "RUN",
            "watch": "SEE",
            "look": "SEE",
            "view": "SEE",
            "observe": "SEE",
            "brother": "SIBLING",
            "sister": "SIBLING",
            "ill": "SICK",
            "unwell": "SICK",
            "ailing": "SICK",
            "talk": "SPEAK",
            "say": "SPEAK",
            "tell": "SPEAK",
            "communicate": "SPEAK",
            "dumb": "STUPID",
            "foolish": "STUPID",
            "idiotic": "STUPID",
            "thanks": "THANK_YOU",
            "thankyou": "THANK_YOU",
            "toss": "THROW",
            "hurl": "THROW",
            "fling": "THROW",
            "period": "TIME",
            "moment": "TIME",
            "abuse": "VIOLENCE",
            "fight": "VIOLENCE",
            "attack": "VIOLENCE",
            "puke": "VOMIT",
            "throw up": "VOMIT",
            "lady": "WOMAN",
            "female": "WOMAN",
            "hate": "I_DISLIKE_YOU",
            "dislike": "I_DISLIKE_YOU",
            "dunno": "I_DONT_KNOW",
            "don't know": "I_DONT_KNOW",
        }

        # Add all direct mappings
        for word, gloss in word_mappings.items():
            mapping[word.lower()] = gloss

        # Add valid glosses mapping to themselves
        for gloss in self.valid_glosses:
            mapping[gloss.lower()] = gloss

        return mapping

    def _map_to_valid_gloss(self, word: str) -> Optional[str]:
        """Map a word to a valid gloss, or return None if no mapping exists"""
        word_lower = word.lower()

        # Check direct mapping
        if word_lower in self.word_to_gloss_map:
            return self.word_to_gloss_map[word_lower]

        # Check if word.upper() is already a valid gloss
        word_upper = word.upper()
        if word_upper in self.valid_glosses:
            return word_upper

        # Try removing common suffixes and check again
        suffixes = ["s", "ed", "ing", "er", "est", "ly"]
        for suffix in suffixes:
            if word_lower.endswith(suffix):
                stem = word_lower[: -len(suffix)]
                if stem in self.word_to_gloss_map:
                    return self.word_to_gloss_map[stem]
                if stem.upper() in self.valid_glosses:
                    return stem.upper()

        return None

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
        """Convert text to NSL glosses (only valid vocabulary)."""
        if not self.nlp:
            raise RuntimeError("spaCy model not loaded. Call load_model() first.")

        doc = self.nlp(text)
        all_clause_glosses = []

        for sent in doc.sents:
            clause_glosses, skipped = self._sentence_to_glosses(sent)
            if clause_glosses:
                all_clause_glosses.append(clause_glosses)
            if self.debug and skipped:
                print(f"Skipped words in sentence: {', '.join(skipped)}")

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

                    compound = "_".join(parts)  # Use underscore for compounds
                    compounds[j] = compound

                    for k in range(i, j):
                        compounds[k] = None

                    i = j + 1
                    continue
            i += 1

        return compounds

    def _separate_digits(self, text: str) -> List[str]:
        """Separate digits in a string into individual digit glosses"""
        digits = []
        for char in text:
            if char.isdigit():
                digits.append(char)
        return digits

    def _get_gloss_token(self, token, compounds_map) -> Optional[str]:
        """Convert a single token to its gloss representation (CONSTRAINED)"""
        if token.i in compounds_map:
            compound = compounds_map[token.i]
            if compound is None:
                return None
            # Check if compound is valid
            if compound in self.valid_glosses:
                return compound
            # Try to map it
            mapped = self._map_to_valid_gloss(compound)
            return mapped

        # Check if token is a number that needs separation
        if token.pos_ == "NUM" and token.text.isdigit() and len(token.text) > 1:
            # Return special marker for multi-digit numbers
            return f"__DIGITS__{token.text}"

        # Get base form
        if token.tag_ in ["PRP$", "WP$"]:
            word = token.text
        elif "'s" in token.text:
            word = token.text.replace("'s", "")
        elif token.pos_ == "NOUN" and token.tag_ in ["NNS", "NNPS"]:
            word = token.text
        else:
            word = token.lemma_

        # Map to valid gloss
        mapped_gloss = self._map_to_valid_gloss(word)
        return mapped_gloss

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
            important_preps = {"with", "after", "before"}
            return token.text.lower() in important_preps

        return False

    def _get_noun_phrase_tokens(self, head_token, compounds_map):
        """
        Get all tokens that are part of a noun phrase.
        Returns list of tokens in correct order, handling compounds.
        """
        phrase_token_set = set()
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

    def _sentence_to_glosses(self, sent) -> Tuple[List[str], List[str]]:
        """
        Convert a sentence to NSL glosses with proper SOV ordering.
        Returns: (glosses, skipped_words)
        """
        glosses = []
        skipped_words = []
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

        # Helper function to add gloss safely
        def add_gloss_safe(token):
            """Add gloss only if not already added AND it's in valid vocabulary"""
            if token in added_tokens:
                return False

            gloss = self._get_gloss_token(token, compounds_map)

            # Handle multi-digit numbers
            if gloss and gloss.startswith("__DIGITS__"):
                digit_string = gloss.replace("__DIGITS__", "")
                for digit in digit_string:
                    if digit in self.valid_glosses and digit not in added_glosses:
                        glosses.append(digit)
                        added_glosses.add(digit)
                        if self.debug:
                            print(f"  ✓ Added digit: {digit}")
                added_tokens.add(token)
                return True

            if gloss and gloss in self.valid_glosses and gloss not in added_glosses:
                glosses.append(gloss)
                added_tokens.add(token)
                added_glosses.add(gloss)
                if self.debug:
                    print(f"  ✓ Added: {token.text} -> {gloss}")
                return True
            elif gloss is None and self._is_content_word(token):
                skipped_words.append(token.text)
                if self.debug:
                    print(f"  ✗ Skipped: {token.text} (no valid mapping)")

            added_tokens.add(token)
            return False

        # Step 2: Time expressions
        time_markers = ["after", "before", "again", "ago", "always"]
        for token in tokens:
            if token.text.lower() in time_markers and self._is_content_word(token):
                add_gloss_safe(token)

        # Step 3: Identify sentence components
        subjects = []
        direct_objects = []
        verbs = []
        modals = []

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

        if self.debug:
            print("\n" + "=" * 70)
            print("SOV REORDERING (Vocabulary Constrained)")
            print("=" * 70)

        # SUBJECTS
        for subj in subjects:
            phrase_tokens = self._get_noun_phrase_tokens(subj, compounds_map)
            for token in phrase_tokens:
                add_gloss_safe(token)

        # DIRECT OBJECTS
        for obj in direct_objects:
            phrase_tokens = self._get_noun_phrase_tokens(obj, compounds_map)
            for token in phrase_tokens:
                add_gloss_safe(token)

        # PREPOSITIONAL PHRASES
        for token in tokens:
            if token.pos_ == "ADP" and self._is_content_word(token):
                if token not in added_tokens:
                    add_gloss_safe(token)
                    # Add object of preposition
                    for child in token.children:
                        if child.dep_ == "pobj":
                            obj_phrase = self._get_noun_phrase_tokens(
                                child, compounds_map
                            )
                            for obj_token in obj_phrase:
                                add_gloss_safe(obj_token)

        # REMAINING CONTENT WORDS
        for token in tokens:
            if self._is_content_word(token) and token not in added_tokens:
                add_gloss_safe(token)

        # VERBS
        for verb in verbs:
            add_gloss_safe(verb)

        if self.debug:
            print()
            print("=" * 70)
            print("NSL GLOSS SEQUENCE (Valid Vocabulary Only)")
            print("=" * 70)
            print(f"Output: {' '.join(glosses)}")
            if skipped_words:
                print(f"Skipped: {', '.join(skipped_words)}")
            print("=" * 70)
            print()

        return glosses, skipped_words


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
