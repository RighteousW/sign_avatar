from ..model_training import load_gloss2text_full_model, gloss2text_translate_sentence
from ..constants import TRAINED_MODELS_DIR
from gtts import gTTS

class Gloss2Text:
    def __init__(self, device, gloss_vocab=None, text_vocab=None):
        model, _gloss_vocab, _text_vocab, _ = load_gloss2text_full_model(
            TRAINED_MODELS_DIR / "gloss2text_best_bleu_synthetic", device
        )
        self.model = model
        self.gloss_vocab = gloss_vocab if gloss_vocab else _gloss_vocab
        self.text_vocab = text_vocab if text_vocab else _text_vocab
        self.device = device

    def infer(self, gloss_sequence: list[str]) -> list[str]:
        translation = gloss2text_translate_sentence(
            self.model, gloss_sequence, self.gloss_vocab, self.text_vocab, self.device
        )
        return translation


class Text2Speech:
    def __init__(self, text: list[str], audio_path: str = "temp_output.mp3"):
        self.text = text
        self.audio_path = audio_path

    def synthesize(self):
        full_text = " ".join(self.text)
        tts = gTTS(text=full_text, lang="en")
        tts.save(self.audio_path)


class Gloss2Audio:
    def __init__(self, device, gloss_vocab=None, text_vocab=None):
        self.gloss2text = Gloss2Text(device, gloss_vocab, text_vocab)

    def infer_and_synthesize(self, gloss_sequence: list[str], audio_path: str = "temp_output.mp3"):
        text = " ".join(self.gloss2text.infer(gloss_sequence))
        tts = Text2Speech(text, audio_path)
        tts.synthesize()
        
        return tts.audio_path

if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gloss_sequence = ["HELLO", "WORLD"]  # Example gloss sequence

    gloss2audio = Gloss2Audio(device)
    audio_file = gloss2audio.infer_and_synthesize(gloss_sequence, "output.mp3")
    print(f"Audio saved to {audio_file}")
