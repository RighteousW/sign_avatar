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

    This converter takes English audio/text and transforms it into NSL gloss notation,
    following NSL's SOV (Subject-Object-Verb) word order and grammar rules.
    """

    def __init__(self, debug=False):
        """
        Initialize the NSL gloss converter.

        Args:
            debug: Set to True to see detailed parsing information (useful for debugging)
        """
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
            # Adjust for ambient noise to improve accuracy
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = self.recognizer.record(source)
        text = self.recognizer.recognize_google(audio)
        return text

    def numpy_to_audio_data(self, audio_array, sample_rate: int) -> sr.AudioData:
        """
        Convert a numpy array to an AudioData object.
        Useful when working with audio from machine learning models.
        """
        import numpy as np
        import wave

        # Flatten if stereo
        if len(audio_array.shape) > 1:
            audio_array = audio_array.flatten()

        # Normalize and convert to 16-bit PCM
        audio_clipped = np.clip(audio_array, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)

        # Write to bytes buffer as WAV
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
        """
        Convert text to Namibian Sign Language glosses.

        Returns: List of clause glosses, where each clause is a list of gloss strings.
                 Example: [['TOMORROW', 'I', 'STORE', 'GO'], ['MILK', 'BUY']]
        """
        if not self.nlp:
            raise RuntimeError("spaCy model not loaded. Call load_model() first.")

        doc = self.nlp(text)
        all_clause_glosses = []

        # Process each sentence
        for sent in doc.sents:
            # Split sentence into clauses
            clauses = self._split_into_clauses(sent)

            # Convert each clause to glosses independently
            for clause in clauses:
                clause_glosses = self._clause_to_glosses(clause)
                if clause_glosses:  # Only add non-empty clauses
                    all_clause_glosses.append(clause_glosses)

        return all_clause_glosses

    def audio_file_to_glosses(
        self, audio_file_path: str
    ) -> Tuple[str, List[List[str]]]:
        """
        Convenience method: convert audio file directly to NSL glosses.
        Returns the recognized text and the list of clause glosses.
        """
        text = self.audio_file_to_text(audio_file_path)
        clause_glosses = self.text_to_glosses(text)
        return text, clause_glosses

    def audio_data_to_glosses(
        self, audio_data: sr.AudioData
    ) -> Tuple[str, List[List[str]]]:
        """
        Convenience method: convert AudioData directly to NSL glosses.
        Returns the recognized text and the list of clause glosses.
        """
        text = self.audio_data_to_text(audio_data)
        clause_glosses = self.text_to_glosses(text)
        return text, clause_glosses

    def numpy_to_glosses(
        self, audio_array, sample_rate: int
    ) -> Tuple[str, List[List[str]]]:
        """
        Convenience method: convert numpy array directly to NSL glosses.
        Returns the recognized text and the list of clause glosses.
        """
        audio_data = self.numpy_to_audio_data(audio_array, sample_rate)
        return self.audio_data_to_glosses(audio_data)

    def _split_into_clauses(self, sent):
        """
        Split a sentence into clauses based on dependency parsing.

        This handles:
        - Subordinate clauses (if/when/because/while/after/before)
        - Coordinating conjunctions (and/but/or)
        - Relative clauses

        Returns a list of token lists, where each token list is a clause.
        """
        tokens = list(sent)

        if self.debug:
            print(f"\n  === Splitting: {sent.text} ===")
            print("  Token dependencies:")
            for token in tokens:
                print(
                    f"    {token.text:15} | POS: {token.pos_:6} | DEP: {token.dep_:10} | HEAD: {token.head.text}"
                )

        # Don't bother splitting really short sentences
        if len(tokens) < 5:
            return [tokens]

        clauses = []

        # Find the main verb (root)
        root_verb = None
        for token in tokens:
            if token.dep_ == "ROOT":
                root_verb = token
                break

        if not root_verb:
            return [tokens]

        # Look for subordinate clause markers
        subordinate_markers = []

        # Type 1: Subordinating conjunctions marked with "mark" dependency
        for i, token in enumerate(tokens):
            if token.dep_ == "mark" and token.text.lower() in [
                "if",
                "when",
                "because",
                "while",
                "after",
                "before",
            ]:
                subordinate_markers.append((i, token))

        # Type 2: Adverbial markers that introduce clauses
        # These are sometimes tagged as "advmod" instead of "mark"
        for i, token in enumerate(tokens):
            if token.dep_ == "advmod" and token.text.lower() in [
                "when",
                "where",
                "while",
            ]:
                # Check if this actually introduces a clause (modifies a verb in advcl)
                if token.head.pos_ == "VERB" and token.head.dep_ == "advcl":
                    subordinate_markers.append((i, token))

        if self.debug and subordinate_markers:
            print(f"  Found {len(subordinate_markers)} subordinate marker(s):")
            for idx, marker in subordinate_markers:
                print(
                    f"    - '{marker.text}' at position {idx}, dep={marker.dep_}, head={marker.head.text}"
                )

        # Split at subordinate clause boundaries
        if subordinate_markers:
            for marker_idx, marker_token in subordinate_markers:
                clause_verb = marker_token.head

                # Gather all tokens that belong to this subordinate clause
                sub_clause_tokens = [marker_token]
                for token in tokens:
                    if token == marker_token:
                        continue
                    if token == clause_verb or self._is_descendant(token, clause_verb):
                        sub_clause_tokens.append(token)

                # Keep tokens in sentence order
                sub_clause_tokens.sort(key=lambda t: t.i)
                if sub_clause_tokens:
                    clauses.append(sub_clause_tokens)

            # Gather main clause tokens (everything not in subordinate clauses)
            sub_clause_token_set = set()
            for clause in clauses:
                sub_clause_token_set.update(clause)

            main_clause_tokens = [t for t in tokens if t not in sub_clause_token_set]
            if main_clause_tokens:
                clauses.append(main_clause_tokens)

            if self.debug:
                print(f"  Split into {len(clauses)} clause(s):")
                for i, clause in enumerate(clauses):
                    clause_text = " ".join([t.text for t in clause])
                    print(f"    Clause {i+1}: {clause_text}")

            return clauses if clauses else [tokens]

        # No subordinate clauses found. Check for coordinating conjunctions.
        for i, token in enumerate(tokens):
            if token.text.lower() in ["and", "but", "or"] and token.dep_ == "cc":
                # Split at the conjunction if it joins two independent clauses
                left_part = tokens[:i]
                right_part = tokens[i + 1 :]

                if left_part and right_part:
                    clauses.append(left_part)
                    clauses.append(right_part)

                    if self.debug:
                        print(f"  Split at coordinating conjunction '{token.text}'")
                        print(f"    Left: {' '.join([t.text for t in left_part])}")
                        print(f"    Right: {' '.join([t.text for t in right_part])}")

                    return clauses

        # No splits needed
        if self.debug:
            print(f"  No splitting needed")
        return [tokens]

    def _is_descendant(self, token, ancestor):
        """Helper function to check if a token is a descendant of another in the parse tree"""
        current = token
        while current.head != current:  # Stop at root
            if current.head == ancestor:
                return True
            current = current.head
        return False

    def _clause_to_glosses(self, clause) -> List[str]:
        """
        Convert a single clause to NSL glosses.

        NSL follows SOV (Subject-Object-Verb) word order with the general structure:
        TIME - LOCATION - SUBJECT - OBJECT - ADJECTIVE - VERB - NEGATION - QUESTION

        Args:
            clause: List of spaCy tokens representing a clause

        Returns:
            List of gloss strings for this clause
        """
        glosses = []
        added_tokens = set()

        # Extract time expressions (both single words and phrases)
        time_words = []
        time_phrases = []
        skip_tokens = set()

        for i, token in enumerate(clause):
            if token in skip_tokens:
                continue

            # Look for time phrases like "next week", "last night"
            if token.text.lower() in ["next", "last", "this"] and i + 1 < len(clause):
                next_token = clause[i + 1]
                if next_token.text.lower() in [
                    "week",
                    "month",
                    "year",
                    "night",
                    "morning",
                    "afternoon",
                    "evening",
                    "day",
                ]:
                    time_phrases.append((token, next_token))
                    skip_tokens.add(token)
                    skip_tokens.add(next_token)
                    continue

            # Single time words
            if token.ent_type_ == "TIME" or token.text.lower() in [
                "now",
                "today",
                "yesterday",
                "tomorrow",
                "soon",
                "later",
                "tonight",
            ]:
                time_words.append(token)

        # Extract location words
        location_words = [t for t in clause if t.ent_type_ in ["GPE", "LOC", "FAC"]]

        # Extract question words (but not when they're subjects/objects in relative clauses)
        question_words = [
            t
            for t in clause
            if t.text.lower() in ["who", "what", "where", "when", "why", "how", "which"]
            and t.dep_ not in ["nsubj", "dobj"]
        ]

        # Extract negations - need to catch all forms
        negations = []
        for token in clause:
            if token.dep_ == "neg":
                negations.append(token)
            elif token.text.lower() in [
                "not",
                "no",
                "never",
                "n't",
            ] and token.pos_ not in ["PRON", "DET"]:
                negations.append(token)
            elif "n't" in token.text.lower() and token.pos_ != "PRON":
                negations.append(token)

        # Also check verb children for negations
        for token in clause:
            if token.pos_ == "VERB":
                for child in token.children:
                    if child.dep_ == "neg" or child.text.lower() in [
                        "not",
                        "n't",
                        "never",
                    ]:
                        if child not in negations:
                            negations.append(child)

        # Extract subjects and objects, keeping complete noun phrases together
        subjects = []
        objects = []
        subject_phrases = []
        object_phrases = []

        for token in clause:
            if token.dep_ in ["nsubj", "nsubjpass"]:
                phrase = self._get_noun_phrase(token)
                subject_phrases.append(phrase)
                subjects.extend(phrase)
            elif token.dep_ in ["dobj", "iobj", "attr"]:
                phrase = self._get_noun_phrase(token)
                object_phrases.append(phrase)
                objects.extend(phrase)
            elif token.dep_ == "pobj" and token.head.dep_ not in ["prep"]:
                # Only include prepositional objects for important prepositions
                if token.head.text.lower() in ["to", "at", "in", "on"]:
                    phrase = self._get_noun_phrase(token)
                    object_phrases.append(phrase)
                    objects.extend(phrase)

        # Extract main verbs (not auxiliaries)
        verbs = [
            t
            for t in clause
            if t.pos_ == "VERB"
            and t.dep_ not in ["aux", "auxpass"]
            and t.lemma_ not in ["be", "have", "do"]
        ]

        # Extract modals and modal auxiliaries
        modals = [
            t
            for t in clause
            if t.tag_ == "MD"
            or (
                t.pos_ == "AUX"
                and t.lemma_ in ["can", "will", "would", "should", "must", "may"]
            )
        ]

        # Extract adjectives and adverbs
        adjectives = [t for t in clause if t.pos_ == "ADJ"]
        verb_adverbs = [t for t in clause if t.pos_ == "ADV" and t.head.pos_ == "VERB"]

        # Now build the glosses in NSL word order

        # 1. TIME comes first
        for time_phrase in time_phrases:
            if (
                time_phrase[0] not in added_tokens
                and time_phrase[1] not in added_tokens
            ):
                glosses.append(time_phrase[0].text.upper())
                glosses.append(time_phrase[1].lemma_.upper())
                added_tokens.add(time_phrase[0])
                added_tokens.add(time_phrase[1])

        for time in time_words:
            if self._keep_token(time) and time not in added_tokens:
                glosses.append(time.lemma_.upper())
                added_tokens.add(time)

        # 2. LOCATION
        for loc in location_words:
            if self._keep_token(loc) and loc not in added_tokens:
                glosses.append(loc.lemma_.upper())
                added_tokens.add(loc)

        # 3. SUBJECT (keep phrases together to preserve possessives etc.)
        for phrase in subject_phrases:
            for subj in phrase:
                if self._keep_token(subj) and subj not in added_tokens:
                    gloss = self._get_lemma_with_possessive(subj)
                    if gloss not in glosses:
                        glosses.append(gloss)
                    added_tokens.add(subj)

        # 4. OBJECT (before verb in SOV)
        for phrase in object_phrases:
            for obj in phrase:
                if self._keep_token(obj) and obj not in added_tokens:
                    gloss = self._get_lemma_with_possessive(obj)
                    if gloss not in glosses:
                        glosses.append(gloss)
                    added_tokens.add(obj)

        # 5. ADJECTIVES
        for adj in adjectives:
            if self._keep_token(adj) and adj not in added_tokens:
                glosses.append(adj.lemma_.upper())
                added_tokens.add(adj)

        # 6. ADVERBS (before verb)
        for adv in verb_adverbs:
            if self._keep_token(adv) and adv not in added_tokens:
                glosses.append(adv.lemma_.upper())
                added_tokens.add(adv)

        # 7. MODALS
        for modal in modals:
            if modal not in added_tokens:
                glosses.append(modal.lemma_.upper())
                added_tokens.add(modal)

        # 8. VERB
        for verb in verbs:
            if self._keep_token(verb) and verb not in added_tokens:
                glosses.append(verb.lemma_.upper())
                added_tokens.add(verb)

        # Catch any remaining content words we might have missed
        processed_tokens = set(time_words + location_words + question_words + negations)
        for phrase in subject_phrases + object_phrases:
            processed_tokens.update(phrase)
        processed_tokens.update(verbs)
        processed_tokens.update(modals)
        processed_tokens.update(adjectives)
        processed_tokens.update(verb_adverbs)

        for token in clause:
            if (
                token not in processed_tokens
                and token not in added_tokens
                and self._keep_token(token)
            ):
                glosses.append(self._get_lemma_with_possessive(token))
                added_tokens.add(token)

        # 9. NEGATION at the end
        for neg in negations:
            if neg not in added_tokens:
                if "n't" in neg.text.lower():
                    if "NOT" not in glosses:
                        glosses.append("NOT")
                    added_tokens.add(neg)
                elif self._keep_token(neg):
                    neg_gloss = neg.lemma_.upper()
                    if neg_gloss in ["NOT", "NEVER", "NO"]:
                        if neg_gloss not in glosses:
                            glosses.append(neg_gloss)
                    added_tokens.add(neg)
                else:
                    # Even if the negation marker doesn't pass _keep_token,
                    # we still want to add NOT to the glosses
                    if neg.text.lower() in ["not", "n't", "never", "no"]:
                        if "NOT" not in glosses and "NEVER" not in glosses:
                            glosses.append("NOT")
                        added_tokens.add(neg)

        # 10. QUESTION words at the very end
        for q in question_words:
            if q not in added_tokens:
                q_gloss = q.lemma_.upper()
                if q_gloss not in glosses:
                    glosses.append(q_gloss)
                added_tokens.add(q)

        return glosses

    def _keep_token(self, token) -> bool:
        """
        Determine if a token should be kept in the gloss output.
        We only keep content words: nouns, verbs, adjectives, adverbs, numbers, pronouns.
        """
        keep_pos = {"NOUN", "VERB", "ADJ", "ADV", "NUM", "PRON", "PROPN"}

        if token.pos_ not in keep_pos:
            return False

        # Filter out function words
        if token.pos_ in {"DET", "ADP", "CCONJ", "SCONJ", "AUX", "PART"}:
            return False

        # Exception: keep modal auxiliaries
        if token.pos_ == "AUX" and token.tag_ == "MD":
            return True

        # Filter out punctuation and whitespace
        if token.is_punct or token.is_space:
            return False

        # Filter out auxiliary verbs
        if token.lemma_ in ["be", "have", "do"] and token.pos_ in ["AUX", "VERB"]:
            return False

        # Filter out expletive "it" (like in "it was cold" or "it is raining")
        if token.text.lower() == "it" and token.pos_ == "PRON":
            if token.dep_ in ["expl", "nsubj"]:
                # Check if it's a weather/impersonal construction
                if token.head.pos_ in ["VERB", "AUX"]:
                    # Weather verbs or copula
                    if token.head.lemma_ in ["be", "rain", "snow", "seem", "appear"]:
                        return False
                    # Copula with adjective (it was cold)
                    for child in token.head.children:
                        if child.pos_ == "ADJ":
                            return False

        return True

    def _get_lemma_with_possessive(self, token) -> str:
        """
        Get the lemma for a token, handling possessives properly.
        """
        # Possessive pronouns stay as-is
        if token.tag_ in ["PRP$", "WP$"]:
            return token.text.upper()

        # Remove possessive markers from nouns
        if "'s" in token.text or "'" in token.text:
            base = token.text.replace("'s", "").replace("'", "")
            return base.upper()

        return token.lemma_.upper()

    def _get_noun_phrase(self, head_token) -> List:
        """
        Extract a complete noun phrase including all modifiers.
        This handles cases like "my brother's car" where we need to keep
        possessives in the right order.
        """
        phrase = []

        # Collect all modifiers with their positions in the sentence
        modifiers = []
        for child in head_token.children:
            if child.dep_ in ["amod", "compound", "nummod"]:
                if self._keep_token(child) or child.tag_ in ["PRP$", "WP$"]:
                    modifiers.append((child.i, child))
            elif child.dep_ == "poss":
                # Handle possessives (including nested ones like "my brother's")
                if self._keep_token(child) or child.tag_ in ["PRP$", "WP$"]:
                    # Check if this possessive has its own modifiers
                    for grandchild in child.children:
                        if grandchild.dep_ == "poss" or grandchild.tag_ in [
                            "PRP$",
                            "WP$",
                        ]:
                            modifiers.append((grandchild.i, grandchild))
                    modifiers.append((child.i, child))

        # Sort by position to maintain the original word order
        modifiers.sort(key=lambda x: x[0])

        # Add all modifiers in order
        for _, mod in modifiers:
            phrase.append(mod)

        # Add the head noun itself
        if self._keep_token(head_token):
            phrase.append(head_token)

        return phrase


# Example usage and testing
if __name__ == "__main__":
    # Create converter with debug mode disabled by default
    converter = AudioToGlossConverter(debug=False)

    if converter.load_model():
        test_sentences = [
            # Basic sentences
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
