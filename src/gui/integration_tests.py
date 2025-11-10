"""
End-to-End System Performance Measurement
Measures complete pipeline performance for both directions:
1. Video → Gloss → Text → Audio (Sign Language to Speech)
2. Audio → Text → Gloss → Avatar (Speech to Sign Language)
"""

import os
import sys
import time
import json
import pickle
import tempfile
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from collections import deque
import speech_recognition as sr
from gtts import gTTS

# Import your modules
try:
    from ..constants import (
        MEDIAPIPE_HAND_LANDMARKER_PATH,
        MEDIAPIPE_POSE_LANDMARKER_PATH,
        get_gesture_metadata_path,
        get_gesture_model_path,
        REPRESENTATIVES_LEFT,
    )
    from ..model_training import GestureRecognizerCNN
    from ..landmark_extraction import LandmarkExtractor
    from ..gloss2audio import Gloss2Text
    from ..audio2gloss import AudioToGlossConverter
    from ..gloss2visualization import GestureTransitionGenerator
except ImportError as e:
    print(f"Import error: {e}")
    print("Run from project root: python3 -m src.gui.integration_tests")
    sys.exit(1)


class EndToEndMeasurement:
    """Measures end-to-end system performance"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {
            "hardware": {
                "device": str(self.device),
                "cpu_model": "AMD Ryzen 5 7520U",
            },
            "video_to_speech": [],
            "speech_to_video": [],
        }

        print("Initializing components...")
        self._load_components()

    def _load_components(self):
        """Load all system components"""

        # 1. Video → Gloss (Gesture Recognition)
        print("  Loading gesture recognition model...")
        with open(str(get_gesture_metadata_path(False, 2)), "rb") as f:
            self.gesture_model_info = pickle.load(f)

        self.gesture_model = GestureRecognizerCNN(
            input_size=self.gesture_model_info["input_size"],
            num_classes=len(self.gesture_model_info["class_names"]),
            hidden_size=self.gesture_model_info["hidden_size"],
            dropout=self.gesture_model_info["dropout"],
        )

        checkpoint = torch.load(
            str(get_gesture_model_path(False, 2)), map_location=self.device
        )
        self.gesture_model.load_state_dict(checkpoint["model_state_dict"])
        self.gesture_model.to(self.device)
        self.gesture_model.eval()

        self.landmark_extractor = LandmarkExtractor(use_pose=True)
        self.feature_info = self.gesture_model_info["feature_info"]
        self.frame_counter = 0
        self.last_processed_features = None

        # 2. Gloss → Text (Translation)
        print("  Loading translation model...")
        self.translator = Gloss2Text(self.device)

        # 3. Audio → Text → Gloss
        print("  Loading audio to gloss converter...")
        self.audio_converter = AudioToGlossConverter()
        if not self.audio_converter.load_model():
            raise RuntimeError("Failed to load spaCy model")

        # 4. Gloss → Avatar
        print("  Loading avatar generator...")
        self.avatar_generator = GestureTransitionGenerator(REPRESENTATIVES_LEFT)

        print("✓ All components loaded\n")

    def extract_features_from_frame_data(self, frame_data: dict) -> np.ndarray:
        """Extract features using same logic as training script"""
        features = []

        # Hand landmarks
        if self.feature_info["hand_landmarks"] > 0:
            max_hands = self.feature_info["max_hands"]
            hand_dim_per_hand = self.feature_info["hand_landmarks_per_hand"]
            hand_features = np.zeros(self.feature_info["hand_landmarks"])

            if frame_data.get("hands"):
                for i, hand_data in enumerate(frame_data["hands"][:max_hands]):
                    if hand_data and "landmarks" in hand_data:
                        landmarks = np.array(hand_data["landmarks"][:21])[:, :3].flatten()
                        start_idx = i * hand_dim_per_hand
                        end_idx = start_idx + len(landmarks)
                        hand_features[start_idx:end_idx] = landmarks
            features.extend(hand_features)

        # Pose landmarks
        if self.feature_info["pose_landmarks"] > 0:
            pose_features = np.zeros(self.feature_info["pose_landmarks"])
            if frame_data.get("pose") and frame_data["pose"]:
                landmarks = np.array(frame_data["pose"]["landmarks"])
                if len(landmarks.shape) > 1 and landmarks.shape[1] > 3:
                    landmarks = landmarks[:, :3]
                landmarks_flat = landmarks.flatten()
                pose_features[: min(len(landmarks_flat), len(pose_features))] = (
                    landmarks_flat[: len(pose_features)]
                )
            features.extend(pose_features)

        return np.array(features, dtype=np.float32)

    def process_frame_with_skip(self, frame):
        """Process frame with 2_skip pattern"""
        frame_index = self.frame_counter
        self.frame_counter += 1

        should_process = frame_index % 3 == 0

        if should_process:
            frame_data = self.landmark_extractor.extract_landmarks_from_frame(frame)
            current_features = self.extract_features_from_frame_data(frame_data)
            self.last_processed_features = current_features
            return [current_features]
        else:
            if self.last_processed_features is not None:
                return [self.last_processed_features.copy()]
            else:
                return [np.zeros(self.feature_info["total_features"], dtype=np.float32)]

    def measure_video_to_speech(self, video_path, ground_truth_glosses=None):
        """
        Measure Video → Gloss → Text → Audio pipeline

        Args:
            video_path: Path to video file
            ground_truth_glosses: Optional list of expected glosses for accuracy
        """
        print(f"\n{'='*60}")
        print(f"Measuring: Video → Gloss → Text → Audio")
        print(f"Video: {video_path.name}")
        print(f"{'='*60}\n")

        result = {
            "video_file": video_path.name,
            "ground_truth_glosses": ground_truth_glosses,
            "stages": {},
            "total_time_ms": 0,
        }

        # Stage 1: Video → Gloss
        start_time = time.perf_counter()
        detected_glosses = self._process_video_to_gloss(video_path)
        stage1_time = (time.perf_counter() - start_time) * 1000

        result["stages"]["video_to_gloss"] = {
            "time_ms": stage1_time,
            "output_glosses": detected_glosses,
            "num_glosses": len(detected_glosses),
        }

        if not detected_glosses:
            result["error"] = "No glosses detected"
            return result

        # Stage 2: Gloss → Text
        start_time = time.perf_counter()
        translated_text = self.translator.infer(detected_glosses)
        translated_text_str = " ".join(translated_text)
        stage2_time = (time.perf_counter() - start_time) * 1000

        result["stages"]["gloss_to_text"] = {
            "time_ms": stage2_time,
            "output_text": translated_text_str,
        }

        # Stage 3: Text → Audio
        start_time = time.perf_counter()
        audio_file = self._text_to_audio(translated_text_str)
        stage3_time = (time.perf_counter() - start_time) * 1000

        result["stages"]["text_to_audio"] = {
            "time_ms": stage3_time,
            "audio_file": audio_file,
        }

        # Calculate total
        result["total_time_ms"] = stage1_time + stage2_time + stage3_time

        # Calculate accuracy if ground truth provided
        if ground_truth_glosses:
            correct = sum(
                1
                for gt, pred in zip(ground_truth_glosses, detected_glosses)
                if gt.lower() == pred.lower()
            )
            result["gloss_accuracy"] = correct / len(ground_truth_glosses)

        self._print_video_to_speech_results(result)
        return result

    def measure_speech_to_video(self, audio_file, ground_truth_text=None):
        """
        Measure Audio → Text → Gloss → Avatar pipeline

        Args:
            audio_file: Path to audio file (WAV)
            ground_truth_text: Optional reference text for accuracy
        """
        print(f"\n{'='*60}")
        print(f"Measuring: Audio → Text → Gloss → Avatar")
        print(
            f"Audio: {audio_file.name if hasattr(audio_file, 'name') else audio_file}"
        )
        print(f"{'='*60}\n")

        result = {
            "audio_file": str(audio_file),
            "ground_truth_text": ground_truth_text,
            "stages": {},
            "total_time_ms": 0,
        }

        # Stage 1: Audio → Text
        start_time = time.perf_counter()
        recognized_text = self._audio_to_text(audio_file)
        stage1_time = (time.perf_counter() - start_time) * 1000

        result["stages"]["audio_to_text"] = {
            "time_ms": stage1_time,
            "output_text": recognized_text,
        }

        if not recognized_text:
            result["error"] = "No text recognized"
            return result

        # Stage 2: Text → Gloss
        start_time = time.perf_counter()
        clause_glosses = self.audio_converter.text_to_glosses(recognized_text)
        glosses = []
        for clause in clause_glosses:
            glosses.extend(clause)
        stage2_time = (time.perf_counter() - start_time) * 1000

        result["stages"]["text_to_gloss"] = {
            "time_ms": stage2_time,
            "output_glosses": glosses,
            "num_glosses": len(glosses),
        }

        # Stage 3: Gloss → Avatar
        start_time = time.perf_counter()
        avatar_file = self._gloss_to_avatar(glosses)
        stage3_time = (time.perf_counter() - start_time) * 1000

        result["stages"]["gloss_to_avatar"] = {
            "time_ms": stage3_time,
            "avatar_file": avatar_file["output_file"],
            "denied_glosses": avatar_file["denied_glosses"],
        }

        # Calculate total
        result["total_time_ms"] = stage1_time + stage2_time + stage3_time

        self._print_speech_to_video_results(result)
        return result

    def _process_video_to_gloss(self, video_path):
        """Process video and detect glosses"""
        detected_glosses = []
        feature_queue = deque(maxlen=self.gesture_model_info["sequence_length"])
        last_prediction = ""
        confidence_threshold = 0.7

        # Reset landmark extractor state
        self.frame_counter = 0
        self.last_processed_features = None

        cap = cv2.VideoCapture(str(video_path))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame (2-skip pattern)
            feature_list = self.process_frame_with_skip(frame)

            for features in feature_list:
                feature_queue.append(features)

                if len(feature_queue) == self.gesture_model_info["sequence_length"]:
                    prediction, confidence = self._run_gesture_inference(feature_queue)

                    if (
                        confidence > confidence_threshold
                        and prediction
                        and prediction != last_prediction
                    ):
                        last_prediction = prediction
                        detected_glosses.append(prediction)

                        # Clear queue after detection
                        for _ in range(
                            int(0.8 * self.gesture_model_info["sequence_length"])
                        ):
                            if feature_queue:
                                feature_queue.popleft()

        cap.release()
        return detected_glosses

    def _run_gesture_inference(self, feature_queue):
        """Run gesture recognition inference"""
        sequence = self._pad_sequence(
            list(feature_queue),
            self.gesture_model_info["sequence_length"],
            self.gesture_model_info["feature_info"]["total_features"],
        )

        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.gesture_model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_class = self.gesture_model_info["class_names"][predicted.item()]
            confidence_value = confidence.item()

        return predicted_class, confidence_value

    def _pad_sequence(self, sequence, target_length, feature_size):
        """Pad or truncate sequence"""
        if len(sequence) > target_length:
            indices = np.linspace(0, len(sequence) - 1, target_length, dtype=int)
            return np.array([sequence[i] for i in indices])
        else:
            padded = list(sequence)
            while len(padded) < target_length:
                padded.append(np.zeros(feature_size))
            return np.array(padded)

    def _text_to_audio(self, text):
        """Convert text to audio file"""
        fd, audio_file = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)

        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(audio_file)

        return audio_file

    def _audio_to_text(self, audio_file):
        """Convert audio to text"""
        recognizer = sr.Recognizer()

        with sr.AudioFile(str(audio_file)) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return ""
        except Exception as e:
            print(f"Speech recognition error: {e}")
            return ""

    def _gloss_to_avatar(self, glosses):
        """Generate avatar visualization"""
        fd, landmark_file = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)

        glosses_lower = [g.lower() for g in glosses]
        result = self.avatar_generator.generate_sequence(
            glosses_lower, 4, landmark_file
        )

        return {
            "output_file": landmark_file,
            "denied_glosses": result["denied_glosses"],
        }

    def _print_video_to_speech_results(self, result):
        """Print formatted results for video→speech"""
        print(f"\nResults: Video → Speech")
        print(f"{'─'*60}")
        print(
            f"Detected Glosses: {', '.join(result['stages']['video_to_gloss']['output_glosses'])}"
        )
        print(f"Translated Text:  {result['stages']['gloss_to_text']['output_text']}")
        print(f"\nTiming Breakdown:")
        print(
            f"  Video → Gloss:  {result['stages']['video_to_gloss']['time_ms']:.2f} ms"
        )
        print(
            f"  Gloss → Text:   {result['stages']['gloss_to_text']['time_ms']:.2f} ms"
        )
        print(
            f"  Text → Audio:   {result['stages']['text_to_audio']['time_ms']:.2f} ms"
        )
        print(f"  {'─'*40}")
        print(f"  Total:          {result['total_time_ms']:.2f} ms")

        if "gloss_accuracy" in result:
            print(f"\nAccuracy: {result['gloss_accuracy']*100:.1f}%")
        print()

    def _print_speech_to_video_results(self, result):
        """Print formatted results for speech→video"""
        print(f"\nResults: Speech → Video")
        print(f"{'─'*60}")
        print(f"Recognized Text: {result['stages']['audio_to_text']['output_text']}")
        print(
            f"Generated Glosses: {', '.join(result['stages']['text_to_gloss']['output_glosses'])}"
        )

        if result["stages"]["gloss_to_avatar"]["denied_glosses"]:
            print(
                f"Warning: Glosses not found in vocabulary: {', '.join(result['stages']['gloss_to_avatar']['denied_glosses'])}"
            )

        print(f"\nTiming Breakdown:")
        print(
            f"  Audio → Text:   {result['stages']['audio_to_text']['time_ms']:.2f} ms"
        )
        print(
            f"  Text → Gloss:   {result['stages']['text_to_gloss']['time_ms']:.2f} ms"
        )
        print(
            f"  Gloss → Avatar: {result['stages']['gloss_to_avatar']['time_ms']:.2f} ms"
        )
        print(f"  {'─'*40}")
        print(f"  Total:          {result['total_time_ms']:.2f} ms")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Measure end-to-end system performance"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="data/demo_videos",
        help="Directory containing test videos",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="data/demo_audio",
        help="Directory containing test audio files",
    )
    parser.add_argument(
        "--output", type=str, default="end_to_end_results.json", help="Output JSON file"
    )

    args = parser.parse_args()

    # Initialize measurement system
    measurement = EndToEndMeasurement()

    # Test Video → Speech direction
    video_dir = Path(args.video_dir)
    if video_dir.exists():
        video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))

        print(f"\n{'='*60}")
        print(f"TESTING VIDEO → SPEECH PIPELINE")
        print(f"Found {len(video_files)} video(s)")
        print(f"{'='*60}")

        for video_path in video_files[:5]:  # Test first 5 videos
            result = measurement.measure_video_to_speech(video_path)
            measurement.results["video_to_speech"].append(result)

    # Test Speech → Video direction
    audio_dir = Path(args.audio_dir)
    if audio_dir.exists():
        audio_files = list(audio_dir.glob("*.wav"))

        print(f"\n{'='*60}")
        print(f"TESTING SPEECH → VIDEO PIPELINE")
        print(f"Found {len(audio_files)} audio file(s)")
        print(f"{'='*60}")

        for audio_path in audio_files[:5]:  # Test first 5 audio files
            result = measurement.measure_speech_to_video(audio_path)
            measurement.results["speech_to_video"].append(result)
    else:
        print(f"\nNote: No audio directory found at {audio_dir}")
        print("To test Speech→Video, create audio files or record some:")
        print("  mkdir -p data/test_audio")
        print("  # Record short English phrases and save as WAV files")

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(measurement.results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")

    # Summary
    if measurement.results["video_to_speech"]:
        avg_time = np.mean(
            [r["total_time_ms"] for r in measurement.results["video_to_speech"]]
        )
        print(f"Video → Speech avg time: {avg_time:.2f} ms")

    if measurement.results["speech_to_video"]:
        avg_time = np.mean(
            [r["total_time_ms"] for r in measurement.results["speech_to_video"]]
        )
        print(f"Speech → Video avg time: {avg_time:.2f} ms")


if __name__ == "__main__":
    main()
