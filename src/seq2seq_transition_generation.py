import os
import pickle
import numpy as np
import torch
import random
from pathlib import Path

from constants import LANDMARKS_DIR, MODELS_TRAINED_DIR, SEQ2SEQ_CONFIG, OUTPUT_DIR
from seq2seq_model_training import LSTMGenerator, LandmarkProcessor


class InferenceVisualizer:
    """Generate transition sequences for visualization"""

    def __init__(self, models_dir=MODELS_TRAINED_DIR):
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load processor and models
        self.processor = self._load_processor()
        self.generator = self._load_generator()

        print(f"Inference visualizer ready on {self.device}")

    def _load_processor(self):
        """Load the trained landmark processor"""
        processor_path = os.path.join(self.models_dir, "processor.pkl")

        if not os.path.exists(processor_path):
            print(f"Warning: Processor not found at {processor_path}")
            print("Creating new processor...")
            processor = LandmarkProcessor()
            processor.fit_scaler_on_sample(LANDMARKS_DIR)
            return processor

        with open(processor_path, "rb") as f:
            processor = pickle.load(f)

        print(f"Loaded processor from {processor_path}")
        return processor

    def _load_generator(self):
        """Load the trained generator model"""
        generator_path = os.path.join(self.models_dir, "lstm_generator.pth")
        config_path = os.path.join(self.models_dir, "gan_config.pkl")

        if not os.path.exists(generator_path):
            raise FileNotFoundError(f"Generator model not found at {generator_path}")

        # Load config
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        else:
            print("Using default config")
            config = SEQ2SEQ_CONFIG

        # Create and load generator
        generator = LSTMGenerator(
            feature_dim=self.processor.total_features,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
        )

        generator.load_state_dict(torch.load(generator_path, map_location=self.device))
        generator.to(self.device)
        generator.eval()

        print(f"Loaded generator from {generator_path}")
        return generator

    def get_landmark_files(self, max_files=200):
        """Get available landmark files"""
        landmark_files = []
        landmarks_path = Path(LANDMARKS_DIR)

        if not landmarks_path.exists():
            raise ValueError(f"Landmarks directory not found: {LANDMARKS_DIR}")

        for folder in landmarks_path.iterdir():
            if folder.is_dir():
                pkl_files = list(folder.glob("*_landmarks.pkl"))
                landmark_files.extend(pkl_files)

        # Limit files for manageable processing
        if len(landmark_files) > max_files:
            landmark_files = random.sample(landmark_files, max_files)

        print(f"Found {len(landmark_files)} landmark files for inference")
        return landmark_files

    def load_file_frames(self, pkl_file):
        """Load and process frames from a landmark file"""
        try:
            with open(pkl_file, "rb") as f:
                landmark_data = pickle.load(f)

            frames = landmark_data.get("frames", [])

            # Convert all frames to features
            processed_frames = []
            for frame_data in frames:
                features = self.processor.extract_frame_features(frame_data)
                normalized = self.processor.normalize_features(features)
                processed_frames.append(normalized)

            return processed_frames, landmark_data

        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            return None, None

    def generate_transition(self, start_frame, end_frame):
        """Generate transition sequence between two frames"""
        with torch.no_grad():
            start_tensor = torch.FloatTensor(start_frame).unsqueeze(0).to(self.device)
            end_tensor = torch.FloatTensor(end_frame).unsqueeze(0).to(self.device)

            # Generate 7-frame sequence (start + 5 generated + end)
            generated_sequence = self.generator(start_tensor, end_tensor)

            return generated_sequence.cpu().numpy()[0]  # Remove batch dimension

    def create_transition_file(
        self,
        file_a_path,
        file_b_path,
        frames_a,
        frames_b,
        landmark_data_a,
        landmark_data_b,
    ):
        """Create transition sequence from file A to file B"""

        # Get 3rd-to-last frame from file A (transition start point)
        if len(frames_a) < 3:
            return None
        start_frame_idx = len(frames_a) - 3
        start_frame = frames_a[start_frame_idx]

        # Get 3rd frame from file B (transition end point)
        if len(frames_b) < 3:
            return None
        end_frame_idx = 2  # 3rd frame (0-indexed)
        end_frame = frames_b[end_frame_idx]

        # Generate transition
        generated_transition = self.generate_transition(start_frame, end_frame)

        # Build complete sequence: start_of_A -> ... -> 3rd_last_A -> generated_transition -> 3rd_B -> ... -> end_of_B
        complete_sequence = []

        # Add frames from A up to and including the transition start
        complete_sequence.extend(frames_a[: start_frame_idx + 1])

        # Add generated transition (excluding start and end frames to avoid duplication)
        complete_sequence.extend(generated_transition[1:-1])  # Only middle 5 frames

        # Add frames from B starting from the transition end
        complete_sequence.extend(frames_b[end_frame_idx:])

        # Create the output data structure
        transition_data = {
            "video_path": f"transition_{file_a_path.stem}_to_{file_b_path.stem}",
            "timestamp": f"generated_transition",
            "frames": [],
            "connections": landmark_data_a.get("connections", {}),
            "feature_info": landmark_data_a.get("feature_info", {}),
            "transition_info": {
                "source_file_a": str(file_a_path),
                "source_file_b": str(file_b_path),
                "transition_start_frame": start_frame_idx,
                "transition_end_frame": end_frame_idx,
                "generated_frames": 5,
                "total_frames": len(complete_sequence),
            },
        }

        # Convert feature vectors back to frame format (this is approximate)
        for i, frame_features in enumerate(complete_sequence):
            # Create a dummy frame structure - your visualization script should handle this
            frame_data = {
                "frame_number": i,
                "hands": [],
                "pose": None,
                "connection_features": {"hands": [], "pose": []},
                "is_generated": start_frame_idx
                < i
                < start_frame_idx + 1 + 5,  # Mark generated frames
                "features": frame_features.tolist(),  # Raw features for debugging
            }
            transition_data["frames"].append(frame_data)

        return transition_data

    def generate_visualizations(self, num_transitions=100, output_dir=OUTPUT_DIR):
        """Generate transition visualizations"""

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        print(f"Creating visualizations in {output_dir}")

        # Get landmark files
        landmark_files = self.get_landmark_files(max_files=150)

        if len(landmark_files) < 2:
            print("Need at least 2 landmark files to create transitions")
            return

        successful_transitions = 0
        attempts = 0

        while (
            successful_transitions < num_transitions and attempts < num_transitions * 3
        ):
            attempts += 1

            # Randomly select two different files
            file_a, file_b = random.sample(landmark_files, 2)

            # Load frames from both files
            frames_a, landmark_data_a = self.load_file_frames(file_a)
            frames_b, landmark_data_b = self.load_file_frames(file_b)

            if frames_a is None or frames_b is None:
                continue

            if len(frames_a) < 5 or len(frames_b) < 5:  # Need minimum frames
                continue

            # Create transition
            try:
                transition_data = self.create_transition_file(
                    file_a, file_b, frames_a, frames_b, landmark_data_a, landmark_data_b
                )

                if transition_data is None:
                    continue

                # Save transition file
                output_filename = f"{file_a.stem}_to_{file_b.stem}_transition.pkl"
                output_path = os.path.join(output_dir, output_filename)

                with open(output_path, "wb") as f:
                    pickle.dump(transition_data, f, protocol=pickle.HIGHEST_PROTOCOL)

                successful_transitions += 1

                if successful_transitions % 10 == 0:
                    print(
                        f"Generated {successful_transitions}/{num_transitions} transitions"
                    )

            except Exception as e:
                print(f"Error creating transition {file_a.stem} -> {file_b.stem}: {e}")
                continue

        print(
            f"\nGenerated {successful_transitions} transition visualizations in {output_dir}"
        )
        print(f"Files are named: [source_file]_to_[target_file]_transition.pkl")

        # Create a summary file
        summary = {
            "total_transitions": successful_transitions,
            "output_directory": output_dir,
            "generated_frames_per_transition": 5,
            "model_used": "LSTM-GAN",
            "timestamp": str(np.datetime64("now")),
            "files": [
                f for f in os.listdir(output_dir) if f.endswith("_transition.pkl")
            ],
        }

        summary_path = os.path.join(output_dir, "transitions_summary.json")
        import json

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return successful_transitions


def main():
    """Generate inference visualizations"""
    print("=== GAN Inference Visualization Generator ===")

    try:
        # Create visualizer
        visualizer = InferenceVisualizer()

        # Generate transitions
        num_generated = visualizer.generate_visualizations(num_transitions=100)

        if num_generated > 0:
            print(f"\n✅ Successfully generated {num_generated} transition sequences!")
            print(f"📁 Output directory: {OUTPUT_DIR}")
            print(
                f"📊 Each file contains: original_sequence + 5_generated_frames + target_sequence"
            )
            print(f"🎬 Use your existing visualization script to view the transitions")
        else:
            print(
                "❌ No transitions were generated. Check your landmark files and trained models."
            )

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Trained models in MODELS_TRAINED_DIR")
        print("2. Landmark files in LANDMARKS_DIR")
        print("3. Proper directory permissions")


if __name__ == "__main__":
    main()
