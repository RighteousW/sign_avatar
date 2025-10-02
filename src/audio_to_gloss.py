"""
Improved Audio to Gloss Converter with better clause splitting and ASL-appropriate rules
"""

import speech_recognition as sr
import spacy
import io
from typing import Tuple, List


class AudioToGlossConverter:
    """Convert audio files to sign language glosses with improved accuracy"""

    def __init__(self, target_language="ASL"):
        """
        Args:
            target_language: "ASL" for American Sign Language or "DGS" for German Sign Language
        """
        self.recognizer = sr.Recognizer()
        self.nlp = None
        self.target_language = target_language

    def load_model(self):
        """Load spaCy language model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            return True
        except OSError:
            print(
                "Error: spaCy model not found. Install with: python -m spacy download en_core_web_sm"
            )
            return False

    def audio_data_to_text(self, audio_data: sr.AudioData) -> str:
        """Convert AudioData object to text using speech recognition"""
        text = self.recognizer.recognize_google(audio_data)
        return text

    def audio_file_to_text(self, audio_file_path: str) -> str:
        """Convert audio file to text using speech recognition"""
        with sr.AudioFile(audio_file_path) as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = self.recognizer.record(source)

        text = self.recognizer.recognize_google(audio)
        return text

    def numpy_to_audio_data(self, audio_array, sample_rate: int) -> sr.AudioData:
        """Convert numpy array to AudioData object"""
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

    def text_to_glosses(self, text: str) -> Tuple[str, List[str]]:
        """
        Convert text to sign language glosses.
        Returns: (glosses_string, glosses_list)
        """
        if not self.nlp:
            raise RuntimeError("spaCy model not loaded. Call load_model() first.")

        doc = self.nlp(text)
        all_glosses = []

        # Process each sentence
        for sent in doc.sents:
            # Split into clauses (improved implementation)
            clauses = self._split_into_clauses(sent)

            for clause in clauses:
                clause_glosses = self._process_clause(clause)
                all_glosses.extend(clause_glosses)

        return " ".join(all_glosses), all_glosses

    def audio_file_to_glosses(self, audio_file_path: str) -> Tuple[str, List[str]]:
        """Convert audio file directly to glosses."""
        text = self.audio_file_to_text(audio_file_path)
        glosses_string, glosses_list = self.text_to_glosses(text)
        return text, glosses_list

    def audio_data_to_glosses(self, audio_data: sr.AudioData) -> Tuple[str, List[str]]:
        """Convert AudioData directly to glosses."""
        text = self.audio_data_to_text(audio_data)
        glosses_string, glosses_list = self.text_to_glosses(text)
        return text, glosses_list

    def numpy_to_glosses(self, audio_array, sample_rate: int) -> Tuple[str, List[str]]:
        """Convert numpy array directly to glosses."""
        audio_data = self.numpy_to_audio_data(audio_array, sample_rate)
        return self.audio_data_to_glosses(audio_data)

    def _split_into_clauses(self, sent):
        """
        Split sentence into clauses based on dependency parsing.
        Handles subordinate clauses (if/when/because), coordinating conjunctions, and relative clauses.
        """
        tokens = list(sent)

        # For simple sentences or very short clauses, don't split
        if len(tokens) < 5:
            return [tokens]

        clauses = []
        main_clause = []
        subordinate_clauses = []

        # Find the root verb
        root_verb = None
        for token in tokens:
            if token.dep_ == "ROOT":
                root_verb = token
                break

        if not root_verb:
            return [tokens]

        # Separate subordinate clauses (if/when/because clauses)
        current_clause = []
        in_subordinate = False
        subordinate_start = None

        for i, token in enumerate(tokens):
            # Check for subordinating conjunctions
            if token.dep_ == "mark" and token.text.lower() in [
                "if",
                "when",
                "because",
                "while",
                "after",
                "before",
            ]:
                if current_clause:
                    main_clause.extend(current_clause)
                    current_clause = []
                in_subordinate = True
                subordinate_start = i
                current_clause.append(token)
            # Check for coordinating conjunctions that split clauses
            elif token.text.lower() in ["and", "but", "or"] and token.dep_ == "cc":
                # Check if this splits two main verbs
                if in_subordinate:
                    current_clause.append(token)
                else:
                    if current_clause:
                        main_clause.extend(current_clause)
                    clauses.append(main_clause)
                    main_clause = []
                    current_clause = []
            else:
                current_clause.append(token)

                # Check if we're ending a subordinate clause (found its verb)
                if in_subordinate and token.pos_ == "VERB" and token.head == root_verb:
                    subordinate_clauses.append(current_clause)
                    current_clause = []
                    in_subordinate = False

        # Add remaining tokens
        if current_clause:
            if in_subordinate:
                subordinate_clauses.append(current_clause)
            else:
                main_clause.extend(current_clause)

        # For conditionals, put subordinate clause first, then main clause
        if subordinate_clauses:
            for sub_clause in subordinate_clauses:
                clauses.append(sub_clause)
            if main_clause:
                clauses.append(main_clause)
        else:
            clauses.append(main_clause if main_clause else tokens)

        return clauses if clauses else [tokens]

    def _process_clause(self, clause) -> List[str]:
        """Process clause according to sign language transformation rules"""
        glosses = []
        added_tokens = set()  # Track tokens we've already added to avoid duplicates

        # Extract components with improved identification
        # Time expressions - keep phrases together
        time_words = []
        time_phrases = []
        skip_tokens = set()

        for i, token in enumerate(clause):
            if token in skip_tokens:
                continue
            # Check for time phrases like "next week", "last night"
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

        location_words = [t for t in clause if t.ent_type_ in ["GPE", "LOC", "FAC"]]

        # Question words (WH-questions go at end in ASL) - but only if they're actually question words, not relative pronouns
        question_words = [
            t
            for t in clause
            if t.text.lower() in ["who", "what", "where", "when", "why", "how", "which"]
            and t.dep_ not in ["nsubj", "dobj"]
        ]  # Exclude if they're subjects/objects in relative clauses

        # Negations (improved detection - check ALL tokens in clause)
        # MUST be defined first before checking verb children
        negations = []
        for token in clause:
            # Check for negation dependency
            if token.dep_ == "neg":
                negations.append(token)
            # Check for negative words (but not "nobody" which is a pronoun subject)
            elif (
                token.text.lower() in ["not", "no", "never", "n't"]
                and token.pos_ != "PRON"
            ):
                negations.append(token)
            # Check for contracted negations (isn't, don't, won't, etc.)
            elif "n't" in token.text.lower() and token.pos_ != "PRON":
                # Extract the "n't" part as negation
                negations.append(token)

        # Subjects and objects (with improved extraction)
        subjects = []
        objects = []
        for token in clause:
            if token.dep_ in ["nsubj", "nsubjpass"]:
                # Get the full noun phrase including possessives
                subjects.extend(self._get_noun_phrase(token))
            elif token.dep_ in ["dobj", "iobj", "attr"]:
                objects.extend(self._get_noun_phrase(token))
            elif token.dep_ == "pobj" and token.head.dep_ not in ["prep"]:
                # Only get prepositional objects if they're important (not just "to the")
                if token.head.text.lower() in ["to", "at", "in", "on"]:
                    objects.extend(self._get_noun_phrase(token))

        # Verbs (exclude auxiliaries)
        verbs = [
            t
            for t in clause
            if t.pos_ == "VERB"
            and t.dep_ not in ["aux", "auxpass"]
            and t.lemma_ not in ["be", "have", "do"]
        ]

        # Check if any verb has a negation child or is part of a contraction
        has_negation = False
        for verb in list(verbs):  # Use list() to avoid modification during iteration
            for child in verb.children:
                if child.dep_ == "neg" or child.text.lower() in ["not", "n't"]:
                    has_negation = True
                    if child not in negations:
                        negations.append(child)

        # Modals and auxiliaries
        modals = [
            t
            for t in clause
            if t.tag_ == "MD"
            or (
                t.pos_ == "AUX"
                and t.lemma_ in ["can", "will", "would", "should", "must", "may"]
            )
        ]

        # Adjectives and adverbs
        adjectives = [t for t in clause if t.pos_ == "ADJ"]
        verb_adverbs = [t for t in clause if t.pos_ == "ADV" and t.head.pos_ == "VERB"]

        # Build glosses based on target language rules
        if self.target_language == "ASL":
            # ASL typically uses: TIME - TOPIC - SUBJECT - OBJECT - VERB - NEGATION - QUESTION

            # Time expressions first (including phrases)
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

            # Location/topic
            for loc in location_words:
                if self._keep_token(loc) and loc not in added_tokens:
                    glosses.append(loc.lemma_.upper())
                    added_tokens.add(loc)

            # Subject
            for subj in subjects:
                if self._keep_token(subj) and subj not in added_tokens:
                    gloss = self._get_lemma_with_possessive(subj)
                    if gloss not in glosses:  # Avoid duplicate glosses
                        glosses.append(gloss)
                        added_tokens.add(subj)

            # Adjectives (before noun they modify or with copula)
            for adj in adjectives:
                if self._keep_token(adj) and adj not in added_tokens:
                    glosses.append(adj.lemma_.upper())
                    added_tokens.add(adj)

            # Object
            for obj in objects:
                if self._keep_token(obj) and obj not in added_tokens:
                    gloss = self._get_lemma_with_possessive(obj)
                    if gloss not in glosses:  # Avoid duplicate glosses
                        glosses.append(gloss)
                        added_tokens.add(obj)

            # Adverbs (before verb)
            for adv in verb_adverbs:
                if self._keep_token(adv) and adv not in added_tokens:
                    glosses.append(adv.lemma_.upper())
                    added_tokens.add(adv)

            # Modals
            for modal in modals:
                if modal not in added_tokens:
                    glosses.append(modal.lemma_.upper())
                    added_tokens.add(modal)

            # Verb
            for verb in verbs:
                if self._keep_token(verb) and verb not in added_tokens:
                    glosses.append(verb.lemma_.upper())
                    added_tokens.add(verb)

        elif self.target_language == "DGS":
            # German Sign Language: Location - Adverb - SOV - Negation

            for loc in location_words:
                if self._keep_token(loc) and loc not in added_tokens:
                    glosses.append(loc.lemma_.upper())
                    added_tokens.add(loc)

            for adv in verb_adverbs:
                if self._keep_token(adv) and adv not in added_tokens:
                    glosses.append(adv.lemma_.upper())
                    added_tokens.add(adv)

            for subj in subjects:
                if self._keep_token(subj) and subj not in added_tokens:
                    glosses.append(self._get_lemma_with_possessive(subj))
                    added_tokens.add(subj)

            for obj in objects:
                if self._keep_token(obj) and obj not in added_tokens:
                    glosses.append(self._get_lemma_with_possessive(obj))
                    added_tokens.add(obj)

            for verb in verbs:
                if self._keep_token(verb) and verb not in added_tokens:
                    glosses.append(verb.lemma_.upper())
                    added_tokens.add(verb)

        # Process remaining content words not yet added
        processed_tokens = set(
            time_words
            + location_words
            + question_words
            + subjects
            + objects
            + verbs
            + modals
            + adjectives
            + verb_adverbs
        )

        for token in clause:
            if (
                token not in processed_tokens
                and token not in added_tokens
                and self._keep_token(token)
            ):
                glosses.append(self._get_lemma_with_possessive(token))
                added_tokens.add(token)

        # Negation at end (common in both ASL and DGS)
        # Only add if we actually found negation markers
        for neg in negations:
            if neg not in added_tokens:
                # For contractions like "can't", just add "NOT"
                if "n't" in neg.text.lower():
                    if "NOT" not in glosses:
                        glosses.append("NOT")
                elif self._keep_token(neg):
                    neg_gloss = neg.lemma_.upper()
                    if neg_gloss == "NOT" or neg_gloss == "NEVER" or neg_gloss == "NO":
                        if neg_gloss not in glosses:
                            glosses.append(neg_gloss)
                added_tokens.add(neg)

        # Question words at end (ASL) - only add once
        if self.target_language == "ASL":
            for q in question_words:
                if q not in added_tokens:
                    q_gloss = q.lemma_.upper()
                    if q_gloss not in glosses:
                        glosses.append(q_gloss)
                    added_tokens.add(q)

        return glosses

    def _keep_token(self, token) -> bool:
        """
        Rule 2 from paper: Keep only content words
        Keep: nouns, verbs, adjectives, adverbs, numerals, pronouns
        """
        keep_pos = {"NOUN", "VERB", "ADJ", "ADV", "NUM", "PRON", "PROPN"}

        if token.pos_ not in keep_pos:
            return False

        # Drop function words
        if token.pos_ in {"DET", "ADP", "CCONJ", "SCONJ", "AUX", "PART"}:
            return False

        # Keep auxiliaries that are modals
        if token.pos_ == "AUX" and token.tag_ == "MD":
            return True

        if token.is_punct or token.is_space:
            return False

        # Drop auxiliary verbs (be, have, do)
        if token.lemma_ in ["be", "have", "do"] and token.pos_ in ["AUX", "VERB"]:
            return False

        return True

    def _get_lemma_with_possessive(self, token) -> str:
        """
        Handle possessive pronouns and nouns properly.
        Convert possessive forms to their base + POSS marker if needed.
        """
        # Check for possessive pronouns
        if token.tag_ in ["PRP$", "WP$"]:  # Possessive pronouns
            return token.text.upper()  # Keep "MY", "YOUR", "HIS", "HER", etc.

        # For possessive nouns (e.g., "John's"), remove the 's
        if "'s" in token.text or "'" in token.text:
            base = token.text.replace("'s", "").replace("'", "")
            return base.upper()

        return token.lemma_.upper()

    def _get_noun_phrase(self, head_token) -> List:
        """
        Extract the full noun phrase including modifiers IN CORRECT ORDER.
        For example, "my big dog" returns [my, big, dog] in that order
        """
        phrase = []

        # First collect all modifiers with their positions
        modifiers = []
        for child in head_token.children:
            if child.dep_ in [
                "amod",
                "compound",
                "poss",
                "nummod",
            ]:  # adjective, compound, possessive, numeral
                if self._keep_token(child) or child.tag_ in [
                    "PRP$",
                    "WP$",
                ]:  # Keep possessive pronouns
                    modifiers.append((child.i, child))  # Store with position

        # Sort modifiers by their position in the sentence to maintain order
        modifiers.sort(key=lambda x: x[0])

        # Add modifiers in order
        for _, mod in modifiers:
            phrase.append(mod)

        # Add the head noun itself
        if self._keep_token(head_token):
            phrase.append(head_token)

        return phrase


# Example usage
if __name__ == "__main__":
    converter = AudioToGlossConverter(target_language="ASL")

    if converter.load_model():
        # Test examples - from simple to complex
        test_sentences = [
            # Basic sentences
            "I am going to the store tomorrow",
            "The dog is not eating his food",
            "She quickly ran to the park",
            # Questions
            "Where is the cat?",
            "What did you eat for breakfast?",
            "When will the meeting start?",
            "Who gave you that book?",
            "Why can't we go outside?",
            "How do you know her name?",
            # Negations (various forms)
            "I don't like vegetables",
            "She will never forget this moment",
            "Nobody came to the party",
            "He hasn't finished his homework yet",
            "They won't be arriving until midnight",
            # Conditionals
            "If it rains, we will stay home",
            "If you study hard, you will pass the exam",
            "When she arrives, we can start dinner",
            # Multiple clauses
            "I went to the store and bought some milk",
            "She was reading a book while he cooked dinner",
            "Because it was cold, I wore my heavy jacket",
            # Possessives
            "My brother's car is very fast",
            "Their teacher gave them a difficult assignment",
            "John's sister lives in New York",
            # Time expressions
            "Yesterday I visited my grandmother",
            "Next week we will travel to Paris",
            "Last night the storm damaged several houses",
            # Complex sentences
            "The young girl who lives next door is learning to play the piano",
            "After finishing his work, he went home and relaxed",
            "If you don't clean your room, you can't watch television tonight",
            "My father, who is a doctor, works at the hospital downtown",
            # Modals and adverbs
            "She must quickly finish her assignment",
            "They should probably leave soon",
            "He can easily solve that problem",
            "We might never see them again",
            # Passive voice
            "The cake was eaten by the children",
            "The letter will be delivered tomorrow",
            # Comparatives and adjectives
            "The red car is faster than the blue one",
            "She is the smartest student in the class",
            "This pizza tastes better than yesterday's",
        ]

        for sentence in test_sentences:
            glosses_str, glosses_list = converter.text_to_glosses(sentence)
            print(f"\nInput: {sentence}")
            print(f"Glosses: {glosses_str}")
