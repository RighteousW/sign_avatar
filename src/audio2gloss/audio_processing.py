from scipy import io
import speech_recognition as sr

def audio_data_to_text(recognizer: sr.Recognizer, audio_data: sr.AudioData) -> str:
    """Convert an AudioData object to text using Google's speech recognition"""
    text = recognizer.recognize_google(audio_data)
    return text

def audio_file_to_text(recognizer, audio_file_path: str) -> str:
    """Convert an audio file (WAV format) to text"""
    with sr.AudioFile(audio_file_path) as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.record(source)
    text = recognizer.recognize_google(audio)
    return text

def numpy_to_audio_data(audio_array, sample_rate: int) -> sr.AudioData:
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
