"""
Audio to Gloss Converter - Extracted from voice2gloss.py
An adaptation of https://arxiv.org/pdf/2305.17714: An Open-Source Gloss-Based Baseline
for Spoken to Signed Language Translation
Standalone module for converting audio files to sign language glosses.
"""

import speech_recognition as sr
import spacy
import io
from typing import Tuple, List


class AudioToGlossConverter:
    """Convert audio files to sign language glosses using the exact voice2gloss logic"""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.nlp = None

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

        # Flatten if stereo and clip to valid range
        if len(audio_array.shape) > 1:
            audio_array = audio_array.flatten()

        # Clip to [-1, 1] range and convert float32 to int16
        audio_clipped = np.clip(audio_array, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)

        # Create WAV in memory
        byte_io = io.BytesIO()
        with wave.open(byte_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 2 bytes for int16
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        # Create AudioData from WAV bytes
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
            clauses = self._split_into_clauses(sent)

            for clause in clauses:
                clause_glosses = self._process_clause(clause)
                all_glosses.extend(clause_glosses)

        return " ".join(all_glosses), all_glosses

    def audio_file_to_glosses(self, audio_file_path: str) -> Tuple[str, List[str]]:
        """
        Convert audio file directly to glosses.

        Returns:
            Tuple of (transcribed_text, glosses_list)
        """
        text = self.audio_file_to_text(audio_file_path)
        glosses_string, glosses_list = self.text_to_glosses(text)
        return text, glosses_list

    def audio_data_to_glosses(self, audio_data: sr.AudioData) -> Tuple[str, List[str]]:
        """
        Convert AudioData directly to glosses (no file I/O).

        Returns:
            Tuple of (transcribed_text, glosses_list)
        """
        text = self.audio_data_to_text(audio_data)
        glosses_string, glosses_list = self.text_to_glosses(text)
        return text, glosses_list

    def numpy_to_glosses(self, audio_array, sample_rate: int) -> Tuple[str, List[str]]:
        """
        Convert numpy array (from sounddevice) directly to glosses.

        Args:
            audio_array: numpy array from sounddevice recording
            sample_rate: sample rate used during recording

        Returns:
            Tuple of (transcribed_text, glosses_list)
        """
        audio_data = self.numpy_to_audio_data(audio_array, sample_rate)
        return self.audio_data_to_glosses(audio_data)

    def _split_into_clauses(self, sent):
        """Simple clause splitting - currently returns whole sentence"""
        return [sent]

    def _process_clause(self, clause) -> List[str]:
        """Process clause according to sign language transformation rules"""
        glosses = []

        # Extract components
        location_words = [t for t in clause if t.ent_type_ in ["GPE", "LOC"]]
        verb_adverbs = [t for t in clause if t.pos_ == "ADV" and t.head.pos_ == "VERB"]
        subjects = [t for t in clause if t.dep_ in ["nsubj", "nsubjpass"]]
        objects = [t for t in clause if t.dep_ in ["dobj", "pobj", "iobj"]]
        verbs = [
            t
            for t in clause
            if t.pos_ == "VERB" and t.dep_ in ["ROOT", "aux", "auxpass", "cop"]
        ]
        negations = [t for t in clause if t.dep_ == "neg"]

        # Rule 4: Location words first
        for loc in location_words:
            if self._keep_token(loc):
                glosses.append(loc.lemma_.upper())

        # Rule 3: Adverbs early
        for adv in verb_adverbs:
            if self._keep_token(adv):
                glosses.append(adv.lemma_.upper())

        # Rule 1: SOV order
        for subj in subjects:
            if self._keep_token(subj):
                glosses.append(subj.lemma_.upper())

        for obj in objects:
            if self._keep_token(obj):
                glosses.append(obj.lemma_.upper())

        for verb in verbs:
            if self._keep_token(verb):
                glosses.append(verb.lemma_.upper())

        # Other content words
        processed_tokens = set(
            location_words + verb_adverbs + subjects + objects + verbs
        )
        for token in clause:
            if token not in processed_tokens and self._keep_token(token):
                glosses.append(token.lemma_.upper())

        # Rule 5: Negation at end
        for neg in negations:
            if self._keep_token(neg):
                glosses.append(neg.lemma_.upper())

        return glosses

    def _keep_token(self, token) -> bool:
        """Rule 2: Keep only content words"""
        keep_pos = {"NOUN", "VERB", "ADJ", "ADV", "NUM", "PRON"}

        if token.pos_ not in keep_pos:
            return False

        if (
            token.pos_ in {"DET", "ADP", "CCONJ", "SCONJ", "AUX"}
            and token.pos_ != "PRON"
        ):
            return False

        if token.is_punct or token.is_space:
            return False

        return True
