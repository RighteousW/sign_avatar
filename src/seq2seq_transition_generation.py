import os
import pickle
import numpy as np
import torch
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import cv2

from constants import LANDMARKS_DIR, MODELS_TRAINED_DIR
from seq2seq_model_training import GestureTransformer, LandmarkProcessor
from landmark_visualization import LandmarkVisualizer


class GestureTransitionCreator:
    """Creates smooth transitions between gesture sequences"""

    def __init__(self, model_dir: str = MODELS_TRAINED_DIR):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load components
        self.processor = self._load_processor()
        self.model = self._load_model()

    def _load_processor(self):
        """Load the saved processor"""
        processor_path = self.model_dir / "processor.pkl"
        with open(processor_path, "rb") as f:
            processor = pickle.load(f)
        print(f"Loaded processor from {processor_path}")
        return processor

    def _load_model(self):
        """Load the trained model"""
        # Load config
        config_path = self.model_dir / "config.json"
        with open(config_path, "r") as f:
            import json

            config = json.load(f)

        # Create model
        model = GestureTransformer(
            feature_dim=config["feature_dim"],
            d_model=config.get("d_model", 256),
            nhead=config.get("nhead", 8),
            num_encoder_layers=config.get("num_encoder_layers", 4),
            num_decoder_layers=config.get("num_decoder_layers", 4),
        )

        # Load weights
        model_path = self.model_dir / "best_model.pth"
        if not model_path.exists():
            model_path = self.model_dir / "final_transformer_model.pth"

        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        print(f"Loaded model from {model_path}")
        return model

    def select_files(self):
        """Select two gesture files"""
        root = tk.Tk()
        root.withdraw()

        first_file = filedialog.askopenfilename(
            title="Select FIRST gesture",
            filetypes=[("Pickle files", "*.pkl")],
            initialdir=LANDMARKS_DIR,
        )

        if not first_file:
            return None, None

        second_file = filedialog.askopenfilename(
            title="Select SECOND gesture",
            filetypes=[("Pickle files", "*.pkl")],
            initialdir=os.path.dirname(first_file),
        )

        root.destroy()
        return first_file, second_file

    def load_gesture(self, file_path: str):
        """Load gesture and extract features"""
        with open(file_path, "rb") as f:
            landmark_data = pickle.load(f)

        # Extract features from all frames
        features = []
        for frame_data in landmark_data["frames"]:
            frame_features = self.processor.extract_features(frame_data)
            normalized = self.processor.normalize_features(frame_features)
            features.append(normalized)

        return np.array(features), landmark_data

    def generate_transition(self, first_seq, second_seq, length=6):
        """Generate transition between sequences"""
        start_frame = torch.FloatTensor(first_seq[-1]).unsqueeze(0).to(self.device)
        end_frame = torch.FloatTensor(second_seq[0]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            transition = self.model.generate_sequence(start_frame, end_frame, length)

        return transition.cpu().numpy().squeeze()

    def create_combined_sequence(
        self, first_seq, transition, second_seq, template_data
    ):
        """Create combined landmark data"""
        # Combine sequences (avoid duplicate frames)
        combined_features = np.concatenate(
            [
                first_seq[:-1],  # All but last
                transition,  # Generated transition
                second_seq[1:],  # All but first
            ]
        )

        # Convert back to landmark format
        combined_data = template_data.copy()
        combined_data["frames"] = []
        combined_data["video_path"] = "generated_transition"

        for i, features in enumerate(combined_features):
            frame_data = {"frame_number": i}

            # Convert features back to hand landmarks only
            frame_data["hands"] = self._features_to_hands(features)

            combined_data["frames"].append(frame_data)

        return combined_data

    def _features_to_hands(self, features):
        """Convert features to hand landmarks (126 features = 2 hands * 21 landmarks * 3 coords)"""
        hands = []

        for hand_idx in range(2):  # Max 2 hands
            start = hand_idx * 63  # 21 landmarks * 3 coords
            end = start + 63

            hand_data = features[start:end]
            landmarks = []

            for lm_idx in range(21):
                lm_start = lm_idx * 3
                lm_end = lm_start + 3
                landmarks.append(hand_data[lm_start:lm_end].tolist())

            # Only add hand if it has non-zero landmarks
            if np.any(np.array(landmarks) != 0):
                hands.append(
                    {
                        "hand_index": hand_idx,
                        "handedness": "Left" if hand_idx == 0 else "Right",
                        "landmarks": landmarks,
                    }
                )

        return hands

    def save_and_visualize(self, combined_data, first_file, second_file):
        """Save combined sequence and launch visualizer"""
        # Create output path
        output_dir = Path(LANDMARKS_DIR) / "transitions"
        output_dir.mkdir(exist_ok=True)

        first_name = Path(first_file).stem.replace("_landmarks", "")
        second_name = Path(second_file).stem.replace("_landmarks", "")
        output_file = output_dir / f"transition_{first_name}_to_{second_name}.pkl"

        # Save
        with open(output_file, "wb") as f:
            pickle.dump(combined_data, f)

        print(f"Saved transition to: {output_file}")
        print(f"Total frames: {len(combined_data['frames'])}")

        # Launch visualizer
        self._launch_visualizer(str(output_file))

        return str(output_file)

    def _launch_visualizer(self, file_path):
        """Launch visualization"""
        visualizer = LandmarkVisualizer()

        print(f"\nShowing transition visualization...")
        print("Controls: SPACE=play/pause, N=next, P=prev, Q=quit")

        # Load the file
        all_data = visualizer.load_landmarks_data([file_path])
        if not all_data:
            print("Could not load visualization data")
            return

        current_frame = 0
        playing = True
        data = all_data[0]["data"]
        total_frames = len(data["frames"])

        while True:
            if total_frames > 0:
                current_frame = max(0, min(current_frame, total_frames - 1))
                frame_data = data["frames"][current_frame]

                vis_frame = visualizer.create_visualization_frame(
                    frame_data,
                    filename=f"Transition - Frame {current_frame + 1}/{total_frames}",
                )

                if playing:
                    cv2.putText(
                        vis_frame,
                        "PLAYING",
                        (500, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("Gesture Transition", vis_frame)

            key = cv2.waitKey(100 if playing else 0) & 0xFF

            if key in [ord("q"), ord("Q"), 27]:
                break
            elif key == ord(" "):
                playing = not playing
            elif key in [ord("n"), ord("N")]:
                current_frame = min(current_frame + 1, total_frames - 1)
            elif key in [ord("p"), ord("P")]:
                current_frame = max(current_frame - 1, 0)

            if playing:
                current_frame = (current_frame + 1) % total_frames

        cv2.destroyAllWindows()

    def create_transition(self, first_file=None, second_file=None, transition_length=6):
        """Main method to create transition"""
        # Select files if not provided
        if not first_file or not second_file:
            first_file, second_file = self.select_files()
            if not first_file or not second_file:
                print("No files selected")
                return

        print(f"Creating transition between:")
        print(f"  First: {os.path.basename(first_file)}")
        print(f"  Second: {os.path.basename(second_file)}")

        # Load gestures
        first_features, first_data = self.load_gesture(first_file)
        second_features, second_data = self.load_gesture(second_file)

        print(f"Loaded sequences:")
        print(f"  First: {first_features.shape[0]} frames")
        print(f"  Second: {second_features.shape[0]} frames")

        # Generate transition
        print(f"Generating {transition_length}-frame transition...")
        transition = self.generate_transition(
            first_features, second_features, transition_length
        )

        # Create combined sequence
        combined_data = self.create_combined_sequence(
            first_features, transition, second_features, first_data
        )

        # Save and visualize
        output_path = self.save_and_visualize(combined_data, first_file, second_file)

        print(f"\nTransition complete!")
        print(f"Combined sequence: {len(combined_data['frames'])} frames")
        print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create gesture transitions")
    parser.add_argument("--first", help="First gesture file")
    parser.add_argument("--second", help="Second gesture file")
    parser.add_argument("--length", type=int, default=6, help="Transition length")

    args = parser.parse_args()

    print("=== Gesture Transition Creator ===")

    try:
        creator = GestureTransitionCreator()
        creator.create_transition(args.first, args.second, args.length)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
