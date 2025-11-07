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
from typing import Tuple, List, Optional

from ..audio2gloss.audio_processing import audio_data_to_text, audio_file_to_text, numpy_to_audio_data
from ..audio2gloss.gloss_converter import text_to_glosses
from ..audio2gloss.word_mapping import _create_word_mapping
from ..audio2gloss.vocabulary import _get_default_vocabulary


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
            self.valid_glosses = _get_default_vocabulary()
        else:
            self.valid_glosses = set(g.upper() for g in valid_glosses)

        # Create mapping for word variations to valid glosses
        self.word_to_gloss_map = _create_word_mapping(self.valid_glosses)

        if self.debug:
            print(f"Loaded {len(self.valid_glosses)} valid glosses")

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

    def audio_file_to_glosses(
        self, audio_file_path: str
    ) -> Tuple[str, List[List[str]]]:
        """Convert audio file directly to NSL glosses"""
        text = audio_file_to_text(self.recognizer, audio_file_path)
        clause_glosses = self.text_to_glosses(text)
        return text, clause_glosses

    def audio_data_to_glosses(
        self, audio_data: sr.AudioData
    ) -> Tuple[str, List[List[str]]]:
        """Convert AudioData directly to NSL glosses"""
        text = audio_data_to_text(self.recognizer, audio_data)
        clause_glosses = self.text_to_glosses(text)
        return text, clause_glosses

    def numpy_to_glosses(
        self, audio_array, sample_rate: int
    ) -> Tuple[str, List[List[str]]]:
        """Convert numpy array directly to NSL glosses"""
        audio_data = numpy_to_audio_data(audio_array, sample_rate)
        return self.audio_data_to_glosses(audio_data)

    # API thingies
    def audio_data_to_text(self, audio_data: sr.AudioData) -> str:
        return audio_data_to_text(self.recognizer, audio_data)

    def audio_file_to_text(self, audio_file_path: str) -> str:
        return audio_file_to_text(self.recognizer, audio_file_path)

    def numpy_to_audio_data(self, audio_array, sample_rate: int) -> sr.AudioData:
        return numpy_to_audio_data(audio_array, sample_rate)
    
    def text_to_glosses(self, text: str) -> List[List[str]]:
        return text_to_glosses(
            text,
            self.nlp,
            self.debug,
            self.valid_glosses,
            self.word_to_gloss_map,
        )


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
