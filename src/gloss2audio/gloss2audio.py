from ..model_training import load_gloss2text_full_model, gloss2text_translate_sentence
from gtts import gTTS


class Gloss2Text:
    def __init__(self, device, gloss_vocab=None, text_vocab=None):
        model, _gloss_vocab, _text_vocab, _ = load_gloss2text_full_model()
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

    def infer_and_synthesize(
        self, gloss_sequence: list[str], audio_path: str = "temp_output.mp3"
    ):
        text = " ".join(self.gloss2text.infer(gloss_sequence))
        tts = Text2Speech(text, audio_path)
        tts.synthesize()

        return tts.audio_path


if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    glosses = [
        "FOR DEFENCE REQUEST IMMUNITY GIUSEPPE GARGANI VOTE",
        "COMPLIANCE WITH OBLIGATIONS FLAG STATES VOTE",
        "CIVIL LIABILITY FINANCIAL GUARANTEES SHIPOWNERS VOTE",
        "SECURITY AT FOOTBALL MATCHES VOTE",
        "FUTURE KOSOVO ROLE EU VOTE",
        "FUTURE EUROPEAN UNION OWN RESOURCES VOTE"
        "IN EUROPE VOTE FUTURE PROFESSIONAL FOOTBALL",
        "IN CAP VOTE INTEGRATION NEW MEMBER STATES",
        "THAT VOTE CONCLUDE",
        "ME GO HOSPITAL TOMORROW"
    ]
    text = [
        "request for defence of the immunity of giuseppe gargani vote",
        "compliance with the obligations of flag states vote",
        "civil liability and financial guarantees of shipowners vote",
        "security at football matches vote",
        "the future of kosovo and the role of the eu vote",
        "the future of the european union's own resources vote",
        "future of professional football in europe vote",
        "the integration of new member states in the cap vote",
        "that concludes the vote .",
        "I am going to the hospital tomorrow"
    ]
    
    for gloss, original_text in list(zip(glosses, text)):
        gloss_sequence = gloss.split(" ")

        gloss2text = Gloss2Text(device)
        text = gloss2text.infer(gloss_sequence)
        print(f"Original text : {original_text}")
        print(f"Glosses       : {gloss}")
        print(f"Predicted text: {" ".join(text)}")
