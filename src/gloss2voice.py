import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import pyttsx3
import spacy


class GlossesToAudioConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Glosses to Audio Converter")
        self.root.geometry("700x700")

        self.tts_engine = None
        self.nlp = None
        self.is_speaking = False

        self.setup_ui()
        self.load_models()

    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self.root, text="Glosses to Audio Converter", font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)

        # Instructions
        instructions = tk.Label(
            self.root,
            text="Enter sign language glosses (uppercase words) to convert to natural speech",
            font=("Arial", 10),
            wraplength=600,
        )
        instructions.pack(pady=5)

        # Example
        example_label = tk.Label(
            self.root,
            text='Example: "QUICKLY STORE I BOOK BUY NOT" → "I quickly buy a book at the store"',
            font=("Arial", 9, "italic"),
            fg="gray",
        )
        example_label.pack(pady=2)

        # Input frame
        input_frame = tk.LabelFrame(
            self.root, text="Input Glosses", font=("Arial", 11, "bold")
        )
        input_frame.pack(pady=10, padx=20, fill="both", expand=True)

        # Glosses input
        self.glosses_input = tk.Text(
            input_frame, wrap=tk.WORD, font=("Arial", 11), height=8
        )
        input_scrollbar = tk.Scrollbar(input_frame)
        self.glosses_input.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        input_scrollbar.pack(side="right", fill="y")
        self.glosses_input.config(yscrollcommand=input_scrollbar.set)
        input_scrollbar.config(command=self.glosses_input.yview)

        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # Load File Button
        load_btn = tk.Button(
            button_frame,
            text="Load Glosses File",
            command=self.load_glosses_file,
            bg="blue",
            fg="white",
            font=("Arial", 11),
            width=18,
            height=2,
        )
        load_btn.pack(side="left", padx=5)

        # Convert Button
        convert_btn = tk.Button(
            button_frame,
            text="Convert to Speech",
            command=self.convert_glosses,
            bg="green",
            fg="white",
            font=("Arial", 11),
            width=18,
            height=2,
        )
        convert_btn.pack(side="left", padx=5)

        # Clear Button
        clear_btn = tk.Button(
            button_frame,
            text="Clear All",
            command=self.clear_all,
            bg="orange",
            fg="white",
            font=("Arial", 11),
            width=18,
            height=2,
        )
        clear_btn.pack(side="left", padx=5)

        # Status Label
        self.status_label = tk.Label(
            self.root, text="Loading models...", font=("Arial", 10), fg="gray"
        )
        self.status_label.pack(pady=5)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, padx=20, fill="both", expand=True)

        # Natural Text Tab
        text_frame = ttk.Frame(self.notebook)
        self.notebook.add(text_frame, text="Natural English")

        self.text_output = tk.Text(text_frame, wrap=tk.WORD, font=("Arial", 11))
        text_scrollbar = tk.Scrollbar(text_frame)
        self.text_output.pack(side="left", fill="both", expand=True)
        text_scrollbar.pack(side="right", fill="y")
        self.text_output.config(yscrollcommand=text_scrollbar.set)
        text_scrollbar.config(command=self.text_output.yview)

        # Analysis Tab
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="Conversion Analysis")

        self.analysis_output = tk.Text(analysis_frame, wrap=tk.WORD, font=("Arial", 9))
        analysis_scrollbar = tk.Scrollbar(analysis_frame)
        self.analysis_output.pack(side="left", fill="both", expand=True)
        analysis_scrollbar.pack(side="right", fill="y")
        self.analysis_output.config(yscrollcommand=analysis_scrollbar.set)
        analysis_scrollbar.config(command=self.analysis_output.yview)

        # Audio controls frame
        audio_frame = tk.Frame(self.root)
        audio_frame.pack(pady=10)

        # Speak Button
        self.speak_btn = tk.Button(
            audio_frame,
            text="▶ Speak Aloud",
            command=self.speak_text,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11),
            width=15,
        )
        self.speak_btn.pack(side="left", padx=5)

        # Stop Button
        self.stop_btn = tk.Button(
            audio_frame,
            text="■ Stop",
            command=self.stop_speaking,
            bg="#f44336",
            fg="white",
            font=("Arial", 11),
            width=15,
            state="disabled",
        )
        self.stop_btn.pack(side="left", padx=5)

        # Copy Button
        copy_btn = tk.Button(
            audio_frame,
            text="Copy Text",
            command=self.copy_text,
            font=("Arial", 11),
            width=15,
        )
        copy_btn.pack(side="left", padx=5)

        # Voice settings frame
        settings_frame = tk.LabelFrame(
            self.root, text="Voice Settings", font=("Arial", 10)
        )
        settings_frame.pack(pady=5, padx=20, fill="x")

        # Speed control
        speed_frame = tk.Frame(settings_frame)
        speed_frame.pack(side="left", padx=10, pady=5)
        tk.Label(speed_frame, text="Speed:", font=("Arial", 9)).pack(side="left")
        self.speed_var = tk.IntVar(value=150)
        speed_scale = tk.Scale(
            speed_frame,
            from_=50,
            to=300,
            orient="horizontal",
            variable=self.speed_var,
            length=150,
        )
        speed_scale.pack(side="left", padx=5)

        # Volume control
        volume_frame = tk.Frame(settings_frame)
        volume_frame.pack(side="left", padx=10, pady=5)
        tk.Label(volume_frame, text="Volume:", font=("Arial", 9)).pack(side="left")
        self.volume_var = tk.DoubleVar(value=1.0)
        volume_scale = tk.Scale(
            volume_frame,
            from_=0.0,
            to=1.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.volume_var,
            length=150,
        )
        volume_scale.pack(side="left", padx=5)

    def load_models(self):
        """Load TTS engine and spaCy model in a separate thread"""
        thread = threading.Thread(target=self._load_models)
        thread.daemon = True
        thread.start()

    def _load_models(self):
        try:
            # Initialize TTS engine
            self.tts_engine = pyttsx3.init()

            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")

            self.root.after(
                0,
                lambda: self.status_label.config(
                    text="Ready - Enter glosses or load a file", fg="green"
                ),
            )
        except Exception as e:
            self.root.after(
                0,
                lambda: self.status_label.config(
                    text=f"Error loading models: {e}", fg="red"
                ),
            )

    def load_glosses_file(self):
        """Load glosses from a text file"""
        file_types = [("Text files", "*.txt"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(
            title="Select Glosses File", filetypes=file_types
        )

        if filename:
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    content = f.read()
                self.glosses_input.delete(1.0, tk.END)
                self.glosses_input.insert(1.0, content)
                self.status_label.config(text=f"Loaded: {filename}", fg="blue")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading file: {e}")

    def convert_glosses(self):
        """Convert glosses to natural English"""
        if not self.nlp or not self.tts_engine:
            messagebox.showerror("Error", "Models not loaded. Please wait or restart.")
            return

        glosses_text = self.glosses_input.get(1.0, tk.END).strip()

        if not glosses_text:
            messagebox.showwarning("No Input", "Please enter glosses to convert.")
            return

        self.status_label.config(
            text="Converting glosses to natural English...", fg="blue"
        )

        # Process in separate thread
        thread = threading.Thread(target=self.process_glosses, args=(glosses_text,))
        thread.daemon = True
        thread.start()

    def process_glosses(self, glosses_text):
        """Process glosses and convert to natural English - REVERSING voice2gloss transformations"""
        try:
            analysis = []
            analysis.append("=" * 60)
            analysis.append("REVERSING VOICE-TO-GLOSS TRANSFORMATIONS")
            analysis.append("=" * 60)
            analysis.append(f"\nInput glosses: {glosses_text}\n")

            # Parse glosses (they're uppercase words)
            glosses = glosses_text.upper().split()
            analysis.append(f"Step 0: Parsed {len(glosses)} gloss tokens")
            analysis.append(f"Tokens: {glosses}\n")

            # Convert to natural English by reversing the transformations
            natural_text = self.glosses_to_natural_english(glosses, analysis)

            # Update UI
            self.root.after(
                0, lambda: self.display_results(natural_text, "\n".join(analysis))
            )

        except Exception as e:
            self.root.after(
                0,
                lambda: messagebox.showerror("Error", f"Error processing glosses: {e}"),
            )
            self.root.after(
                0, lambda: self.status_label.config(text="Conversion failed", fg="red")
            )

    def glosses_to_natural_english(self, glosses, analysis):
        """
        Reverse the voice2gloss transformations:
        Original order: Location → Adverbs → Subject → Object → Verb → Negation
        Target order: Subject → Adverbs → Verb → Object → Location (with negation inserted)
        """

        if not glosses:
            return "Please enter valid glosses."

        # Step 1: Identify and separate components based on voice2gloss rules
        analysis.append("Step 1: REVERSE Rule 5 - Extract negation from end")

        negation = []
        # Check last tokens for negation
        negation_words = ["NOT", "NO", "NEVER", "NONE"]
        while glosses and glosses[-1] in negation_words:
            negation.insert(0, glosses.pop())

        analysis.append(f"  Negation found: {negation if negation else 'None'}")
        analysis.append(f"  Remaining: {glosses}\n")

        # Step 2: Identify location words at the start (Rule 4 reversal)
        analysis.append("Step 2: REVERSE Rule 4 - Extract location from start")

        locations = []
        location_indicators = [
            "STORE",
            "HOME",
            "SCHOOL",
            "PARK",
            "WORK",
            "HOSPITAL",
            "LIBRARY",
            "MARKET",
            "CHURCH",
            "RESTAURANT",
            "OFFICE",
            "HERE",
            "THERE",
            "WHERE",
        ]

        while glosses and glosses[0] in location_indicators:
            locations.append(glosses.pop(0))

        analysis.append(f"  Locations found: {locations if locations else 'None'}")
        analysis.append(f"  Remaining: {glosses}\n")

        # Step 3: Identify adverbs at the start (Rule 3 reversal)
        analysis.append("Step 3: REVERSE Rule 3 - Extract adverbs from start")

        adverbs = []
        adverb_words = [
            "QUICKLY",
            "SLOWLY",
            "CAREFULLY",
            "FAST",
            "WELL",
            "BADLY",
            "ALWAYS",
            "OFTEN",
            "SOMETIMES",
            "RARELY",
            "USUALLY",
            "REALLY",
            "VERY",
            "TOO",
            "QUITE",
            "EXTREMELY",
        ]

        while glosses and glosses[0] in adverb_words:
            adverbs.append(glosses.pop(0))

        analysis.append(f"  Adverbs found: {adverbs if adverbs else 'None'}")
        analysis.append(f"  Remaining: {glosses}\n")

        # Step 4: Reverse SOV to SVO (Rule 1 reversal)
        analysis.append("Step 4: REVERSE Rule 1 - Convert SOV back to SVO")

        # Identify Subject, Object, Verb
        # In SOV order: Subject comes first, Verb comes last, Object in middle
        subject = None
        verb = None
        objects = []

        if len(glosses) >= 1:
            # First token is likely subject
            subject = glosses[0]

            if len(glosses) >= 2:
                # Last token is likely verb (since it's SOV order)
                verb = glosses[-1]

                # Everything in between is object(s)
                if len(glosses) > 2:
                    objects = glosses[1:-1]

        analysis.append(f"  Subject: {subject}")
        analysis.append(f"  Object(s): {objects}")
        analysis.append(f"  Verb: {verb}\n")

        # Step 5: Reconstruct natural English (reverse Rule 2 - add back function words)
        analysis.append(
            "Step 5: REVERSE Rule 2 - Add back articles, prepositions, conjugations"
        )

        sentence_parts = []

        # Add subject with article if needed
        if subject:
            subject_lower = subject.lower()

            # Pronouns don't need articles
            if subject_lower in ["i", "you", "he", "she", "it", "we", "they"]:
                sentence_parts.append(subject_lower)
            else:
                # Add article for nouns
                sentence_parts.append(self.add_article(subject_lower))

        # Add adverbs after subject
        if adverbs:
            sentence_parts.extend([adv.lower() for adv in adverbs])

        # Add negation and verb
        if negation and verb:
            # Add auxiliary verb for negation
            if subject and subject.lower() in ["he", "she", "it"]:
                sentence_parts.append("does not" if "NOT" in negation else "did not")
                sentence_parts.append(verb.lower())  # Use base form with auxiliary
            else:
                sentence_parts.append("do not" if "NOT" in negation else "did not")
                sentence_parts.append(verb.lower())
        elif verb:
            # Conjugate verb if no negation
            verb_lower = verb.lower()
            if subject and subject.lower() in ["he", "she", "it"]:
                verb_lower = self.conjugate_third_person(verb_lower)
            sentence_parts.append(verb_lower)

        # Add objects with articles
        for obj in objects:
            obj_lower = obj.lower()
            if obj_lower not in ["i", "you", "he", "she", "it", "we", "they"]:
                sentence_parts.append(self.add_article(obj_lower))
            else:
                sentence_parts.append(obj_lower)

        # Add location with preposition
        if locations:
            for loc in locations:
                loc_lower = loc.lower()
                prep = self.get_location_preposition(loc_lower)
                if loc_lower == "here":
                    sentence_parts.append("here")
                elif loc_lower == "there":
                    sentence_parts.append("there")
                else:
                    sentence_parts.append(f"{prep} the {loc_lower}")

        analysis.append(f"  Reconstructed parts: {sentence_parts}\n")

        # Join and capitalize
        if sentence_parts:
            natural_text = " ".join(sentence_parts)
            natural_text = natural_text[0].upper() + natural_text[1:] + "."
        else:
            natural_text = "Could not parse glosses."

        analysis.append(f"Step 6: Final sentence: {natural_text}")

        return natural_text

    def add_article(self, word):
        """Add appropriate article to a noun"""
        vowels = "aeiou"
        if word[0] in vowels:
            return f"an {word}"
        return f"a {word}"

    def conjugate_third_person(self, verb):
        """Simple third person singular conjugation"""
        irregular = {
            "go": "goes",
            "do": "does",
            "have": "has",
            "be": "is",
            "say": "says",
        }

        if verb in irregular:
            return irregular[verb]
        elif verb.endswith(("s", "ss", "sh", "ch", "x", "z", "o")):
            return verb + "es"
        elif verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
            return verb[:-1] + "ies"
        else:
            return verb + "s"

    def get_location_preposition(self, location):
        """Get appropriate preposition for location"""
        at_locations = [
            "home",
            "school",
            "work",
            "store",
            "hospital",
            "library",
            "church",
            "office",
            "park",
        ]

        if location in at_locations:
            return "at"
        return "to"

    def display_results(self, natural_text, analysis):
        """Display conversion results"""
        self.text_output.delete(1.0, tk.END)
        self.analysis_output.delete(1.0, tk.END)

        self.text_output.insert(tk.END, natural_text)
        self.analysis_output.insert(tk.END, analysis)

        self.status_label.config(text="Conversion completed successfully!", fg="green")

    def speak_text(self):
        """Speak the converted text aloud"""
        if not self.tts_engine:
            messagebox.showerror("Error", "TTS engine not initialized.")
            return

        text = self.text_output.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning(
                "No Text", "No text to speak. Convert glosses first."
            )
            return

        # Update button states
        self.speak_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.is_speaking = True

        # Speak in separate thread
        thread = threading.Thread(target=self._speak, args=(text,))
        thread.daemon = True
        thread.start()

    def _speak(self, text):
        """Internal method to speak text"""
        try:
            # Set voice properties
            self.tts_engine.setProperty("rate", self.speed_var.get())
            self.tts_engine.setProperty("volume", self.volume_var.get())

            # Speak
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

        except Exception as e:
            self.root.after(
                0, lambda: messagebox.showerror("Speech Error", f"Error speaking: {e}")
            )
        finally:
            self.is_speaking = False
            self.root.after(0, self._reset_audio_buttons)

    def stop_speaking(self):
        """Stop speaking"""
        if self.tts_engine and self.is_speaking:
            self.tts_engine.stop()
            self.is_speaking = False
            self._reset_audio_buttons()

    def _reset_audio_buttons(self):
        """Reset audio button states"""
        self.speak_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def copy_text(self):
        """Copy converted text to clipboard"""
        text = self.text_output.get(1.0, tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Copied", "Text copied to clipboard!")
        else:
            messagebox.showwarning("No Text", "No text to copy!")

    def clear_all(self):
        """Clear all text fields"""
        self.glosses_input.delete(1.0, tk.END)
        self.text_output.delete(1.0, tk.END)
        self.analysis_output.delete(1.0, tk.END)
        self.status_label.config(text="Cleared - Ready for new input", fg="blue")


def main():
    root = tk.Tk()
    _ = GlossesToAudioConverter(root)
    root.mainloop()


if __name__ == "__main__":
    main()
