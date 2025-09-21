import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import speech_recognition as sr
import threading
import os
import spacy
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav


class AudioToGlossesConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio to Sign Language Glosses Converter")
        self.root.geometry("600x600")

        self.recognizer = sr.Recognizer()
        self.nlp = None
        self.is_recording = False
        self.recorded_audio = None
        self.sample_rate = 44100

        self.setup_ui()
        self.load_spacy_model()

    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self.root, text="Audio to Sign Language Glosses", font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)

        # Instructions
        instructions = tk.Label(
            self.root,
            text="Record live audio or select an audio file (WAV, FLAC, AIFF) to convert to sign language glosses",
            font=("Arial", 10),
            wraplength=500,
        )
        instructions.pack(pady=5)

        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # Record Audio Button
        self.record_btn = tk.Button(
            button_frame,
            text="Start Recording",
            command=self.toggle_recording,
            bg="green",
            fg="white",
            font=("Arial", 12),
            width=20,
            height=2,
        )
        self.record_btn.pack(side="left", padx=10)

        # Choose File Button
        choose_btn = tk.Button(
            button_frame,
            text="Choose Audio File",
            command=self.choose_file,
            bg="blue",
            fg="white",
            font=("Arial", 12),
            width=20,
            height=2,
        )
        choose_btn.pack(side="left", padx=10)

        # Status Label
        self.status_label = tk.Label(
            self.root, text="Loading spaCy model...", font=("Arial", 10), fg="gray"
        )
        self.status_label.pack(pady=5)

        # Selected file label
        self.file_label = tk.Label(
            self.root, text="No file selected", font=("Arial", 9), fg="gray"
        )
        self.file_label.pack(pady=5)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, padx=20, fill="both", expand=True)

        # Original Text Tab
        text_frame = ttk.Frame(self.notebook)
        self.notebook.add(text_frame, text="Original Text")

        self.text_output = tk.Text(text_frame, wrap=tk.WORD, font=("Arial", 10))
        text_scrollbar = tk.Scrollbar(text_frame)
        self.text_output.pack(side="left", fill="both", expand=True)
        text_scrollbar.pack(side="right", fill="y")
        self.text_output.config(yscrollcommand=text_scrollbar.set)
        text_scrollbar.config(command=self.text_output.yview)

        # Glosses Tab
        glosses_frame = ttk.Frame(self.notebook)
        self.notebook.add(glosses_frame, text="Sign Language Glosses")

        self.glosses_output = tk.Text(
            glosses_frame, wrap=tk.WORD, font=("Arial", 10, "bold")
        )
        glosses_scrollbar = tk.Scrollbar(glosses_frame)
        self.glosses_output.pack(side="left", fill="both", expand=True)
        glosses_scrollbar.pack(side="right", fill="y")
        self.glosses_output.config(yscrollcommand=glosses_scrollbar.set)
        glosses_scrollbar.config(command=self.glosses_output.yview)

        # Analysis Tab
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="Linguistic Analysis")

        self.analysis_output = tk.Text(analysis_frame, wrap=tk.WORD, font=("Arial", 9))
        analysis_scrollbar = tk.Scrollbar(analysis_frame)
        self.analysis_output.pack(side="left", fill="both", expand=True)
        analysis_scrollbar.pack(side="right", fill="y")
        self.analysis_output.config(yscrollcommand=analysis_scrollbar.set)
        analysis_scrollbar.config(command=self.analysis_output.yview)

        # Buttons frame
        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(pady=10)

        # Copy buttons
        copy_text_btn = tk.Button(
            buttons_frame,
            text="Copy Original Text",
            command=lambda: self.copy_text(self.text_output),
            font=("Arial", 9),
        )
        copy_text_btn.pack(side="left", padx=5)

        copy_glosses_btn = tk.Button(
            buttons_frame,
            text="Copy Glosses",
            command=lambda: self.copy_text(self.glosses_output),
            font=("Arial", 9),
        )
        copy_glosses_btn.pack(side="left", padx=5)

    def load_spacy_model(self):
        """Load spaCy model in a separate thread"""
        thread = threading.Thread(target=self._load_model)
        thread.daemon = True
        thread.start()

    def _load_model(self):
        try:
            # Try to load English model
            self.nlp = spacy.load("en_core_web_sm")
            self.root.after(
                0,
                lambda: self.status_label.config(
                    text="Ready - Select an audio file", fg="green"
                ),
            )
        except OSError:
            self.root.after(
                0,
                lambda: self.status_label.config(
                    text="Please install: python -m spacy download en_core_web_sm",
                    fg="red",
                ),
            )

    def toggle_recording(self):
        """Toggle between start and stop recording"""
        if not self.nlp:
            messagebox.showerror(
                "Error", "spaCy model not loaded. Please install en_core_web_sm model."
            )
            return

        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self.record_btn.config(text="Stop Recording", bg="red")
        self.status_label.config(
            text="Recording... Click 'Stop Recording' when done", fg="red"
        )
        self.file_label.config(text="Recording in progress...")

        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()

    def record_audio(self):
        """Record audio using sounddevice"""
        try:
            # Record audio
            duration = 30  # Maximum recording duration in seconds
            self.recorded_audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
            )

            # Wait for recording to finish or stop button to be pressed
            start_time = 0
            while self.is_recording and start_time < duration:
                sd.sleep(100)  # Sleep for 100ms
                start_time += 0.1

            # Stop recording
            sd.stop()

        except Exception as e:
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Recording Error", f"Error during recording: {e}"
                ),
            )

    def stop_recording(self):
        """Stop recording and process the audio"""
        self.is_recording = False
        sd.stop()  # Stop the recording
        self.record_btn.config(text="Start Recording", bg="green")
        self.status_label.config(text="Processing recording...", fg="blue")
        self.file_label.config(text="Processing recorded audio...")

        # Process the recording in a separate thread
        thread = threading.Thread(target=self.process_recorded_audio)
        thread.daemon = True
        thread.start()

    def process_recorded_audio(self):
        """Process the recorded audio"""
        try:
            if self.recorded_audio is None:
                raise Exception("No audio was recorded")

            # Save recorded audio to temporary file
            temp_filename = "temp_recording.wav"

            # Convert float32 to int16 for WAV format
            audio_int16 = (self.recorded_audio * 32767).astype(np.int16)

            # Save the recorded audio
            wav.write(temp_filename, self.sample_rate, audio_int16)

            # Process the audio file
            with sr.AudioFile(temp_filename) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Record the audio
                audio = self.recognizer.record(source)

            # Update status
            self.root.after(
                0,
                lambda: self.status_label.config(
                    text="Converting speech to text...", fg="blue"
                ),
            )

            # Recognize speech
            text = self.recognizer.recognize_google(audio)

            # Update status
            self.root.after(
                0,
                lambda: self.status_label.config(
                    text="Converting text to sign language glosses...", fg="blue"
                ),
            )

            # Convert to glosses
            glosses, analysis = self.text_to_glosses(text)

            # Update UI with results
            self.root.after(
                0, lambda: self.display_results(text, glosses, analysis, "success")
            )

        except sr.UnknownValueError:
            self.root.after(
                0,
                lambda: self.display_results(
                    "Could not understand the recorded audio. Please try speaking more clearly.",
                    "",
                    "",
                    "error",
                ),
            )
        except sr.RequestError as e:
            self.root.after(
                0,
                lambda: self.display_results(
                    f"Error with speech recognition service: {e}", "", "", "error"
                ),
            )
        except Exception as e:
            self.root.after(
                0,
                lambda: self.display_results(
                    f"Error processing recording: {e}", "", "", "error"
                ),
            )
        finally:
            # Clean up temporary file
            temp_filename = "temp_recording.wav"
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def choose_file(self):
        if not self.nlp:
            messagebox.showerror(
                "Error", "spaCy model not loaded. Please install en_core_web_sm model."
            )
            return

        file_types = [
            ("Audio files", "*.wav *.flac *.aiff"),
            ("WAV files", "*.wav"),
            ("FLAC files", "*.flac"),
            ("AIFF files", "*.aiff"),
            ("All files", "*.*"),
        ]

        filename = filedialog.askopenfilename(
            title="Select Audio File", filetypes=file_types
        )

        if filename:
            self.file_label.config(text=f"Selected: {os.path.basename(filename)}")
            self.status_label.config(text="Processing audio file...", fg="blue")

            # Process in separate thread to avoid freezing UI
            thread = threading.Thread(target=self.process_audio_file, args=(filename,))
            thread.daemon = True
            thread.start()

    def process_audio_file(self, filename):
        try:
            # Load audio file
            with sr.AudioFile(filename) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Record the audio
                audio = self.recognizer.record(source)

            # Update status
            self.root.after(
                0,
                lambda: self.status_label.config(
                    text="Converting speech to text...", fg="blue"
                ),
            )

            # Recognize speech using Google's speech recognition
            text = self.recognizer.recognize_google(audio)

            # Update status
            self.root.after(
                0,
                lambda: self.status_label.config(
                    text="Converting text to sign language glosses...", fg="blue"
                ),
            )

            # Convert to glosses
            glosses, analysis = self.text_to_glosses(text)

            # Update UI with results
            self.root.after(
                0, lambda: self.display_results(text, glosses, analysis, "success")
            )

        except sr.UnknownValueError:
            self.root.after(
                0,
                lambda: self.display_results(
                    "Could not understand the audio. Please try with clearer speech.",
                    "",
                    "",
                    "error",
                ),
            )
        except sr.RequestError as e:
            self.root.after(
                0,
                lambda: self.display_results(
                    f"Error with speech recognition service: {e}", "", "", "error"
                ),
            )
        except Exception as e:
            self.root.after(
                0,
                lambda: self.display_results(
                    f"Error processing file: {e}", "", "", "error"
                ),
            )

    def text_to_glosses(self, text):
        """Convert text to sign language glosses using spaCy-based approach"""
        if not self.nlp:
            return "Error: spaCy model not loaded", ""

        doc = self.nlp(text)
        analysis_parts = []
        all_glosses = []

        # Process each sentence
        for sent in doc.sents:
            analysis_parts.append(f"SENTENCE: {sent.text}")

            # Split into clauses (simplified - just use commas and conjunctions)
            clauses = self.split_into_clauses(sent)

            sentence_glosses = []
            for clause in clauses:
                clause_glosses, clause_analysis = self.process_clause(clause)
                sentence_glosses.extend(clause_glosses)
                analysis_parts.extend(clause_analysis)

            all_glosses.extend(sentence_glosses)
            analysis_parts.append("")  # Empty line between sentences

        return " ".join(all_glosses), "\n".join(analysis_parts)

    def split_into_clauses(self, sent):
        """Simple clause splitting based on punctuation and conjunctions"""
        # For now, just return the whole sentence as one clause
        # In a more sophisticated implementation, you'd split on commas, conjunctions, etc.
        return [sent]

    def process_clause(self, clause):
        """Process a clause according to the sign language transformation rules"""
        analysis = [f"Processing clause: {clause.text}"]

        # Find SVO triplets
        subjects = []
        verbs = []
        objects = []
        other_tokens = []

        for token in clause:
            analysis.append(
                f"Token: {token.text} | POS: {token.pos_} | DEP: {token.dep_} | HEAD: {token.head.text}"
            )

            if token.dep_ in ["nsubj", "nsubjpass"]:
                subjects.append(token)
            elif token.pos_ == "VERB" and token.dep_ in [
                "ROOT",
                "aux",
                "auxpass",
                "cop",
            ]:
                verbs.append(token)
            elif token.dep_ in ["dobj", "pobj", "iobj"]:
                objects.append(token)
            else:
                other_tokens.append(token)

        # Apply transformation rules
        glosses = []

        # Rule 4: Move location words to start
        location_words = [t for t in clause if t.ent_type_ in ["GPE", "LOC"]]
        for loc in location_words:
            if self.keep_token(loc):
                glosses.append(loc.lemma_.upper())

        # Rule 3: Move verb-modifying adverbs to start
        verb_adverbs = [t for t in clause if t.pos_ == "ADV" and t.head.pos_ == "VERB"]
        for adv in verb_adverbs:
            if self.keep_token(adv):
                glosses.append(adv.lemma_.upper())

        # Rule 1: Reorder SVO to SOV
        # Add subjects
        for subj in subjects:
            if self.keep_token(subj):
                glosses.append(subj.lemma_.upper())

        # Add objects before verbs
        for obj in objects:
            if self.keep_token(obj):
                glosses.append(obj.lemma_.upper())

        # Add verbs
        for verb in verbs:
            if self.keep_token(verb):
                glosses.append(verb.lemma_.upper())

        # Add other content words (except those already processed)
        processed_tokens = set(
            location_words + verb_adverbs + subjects + objects + verbs
        )
        for token in clause:
            if token not in processed_tokens and self.keep_token(token):
                glosses.append(token.lemma_.upper())

        # Rule 5: Move negation to end
        negation_words = [t for t in clause if t.dep_ == "neg"]
        for neg in negation_words:
            if self.keep_token(neg):
                glosses.append(neg.lemma_.upper())

        analysis.append(f"Generated glosses: {' '.join(glosses)}")
        analysis.append("")

        return glosses, analysis

    def keep_token(self, token):
        """Rule 2: Keep only content words"""
        keep_pos = {"NOUN", "VERB", "ADJ", "ADV", "NUM", "PRON"}

        # Skip if not content word
        if token.pos_ not in keep_pos:
            return False

        # Skip articles, prepositions, etc.
        if (
            token.pos_ in {"DET", "ADP", "CCONJ", "SCONJ", "AUX"}
            and token.pos_ != "PRON"
        ):
            return False

        # Skip punctuation and spaces
        if token.is_punct or token.is_space:
            return False

        return True

    def display_results(self, text, glosses, analysis, result_type):
        # Clear previous content
        self.text_output.delete(1.0, tk.END)
        self.glosses_output.delete(1.0, tk.END)
        self.analysis_output.delete(1.0, tk.END)

        if result_type == "success":
            # Insert original text
            self.text_output.insert(tk.END, text)

            # Insert glosses
            self.glosses_output.insert(tk.END, glosses)

            # Insert analysis
            self.analysis_output.insert(tk.END, analysis)

            self.status_label.config(
                text="Conversion completed successfully!", fg="green"
            )
        else:
            self.text_output.insert(tk.END, text)  # Error message
            self.status_label.config(text="Conversion failed", fg="red")

    def copy_text(self, text_widget):
        text = text_widget.get(1.0, tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Copied", "Text copied to clipboard!")
        else:
            messagebox.showwarning("No Text", "No text to copy!")


def main():
    root = tk.Tk()
    app = AudioToGlossesConverter(root)
    root.mainloop()


if __name__ == "__main__":
    main()
