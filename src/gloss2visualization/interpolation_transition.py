import os
import pickle
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
from collections import defaultdict
import re
from datetime import datetime
import random

from ..constants import GESTURE_MODEL_2_SKIP_METADATA_PATH, LANDMARKS_DIR_HANDS_ONLY, OUTPUT_DIR, REPRESENTATIVES_LEFT


class GestureRepresentativeSelector:
    """Selects and stores representative gesture files"""

    def extract_gloss_from_filename(self, filename: str) -> str:
        """Extract gloss from filename by removing timestamp and suffixes"""
        base_name = filename.replace("_landmarks.pkl", "")

        if base_name.endswith("_flipped"):
            base_name = base_name[:-8]

        timestamp_pattern = r"_(\d{8}_\d{6})$"
        match = re.search(timestamp_pattern, base_name)

        if match:
            gloss = base_name[: match.start()]
        else:
            parts = base_name.split("_")
            if len(parts) >= 3:
                gloss = "_".join(parts[:-2])
            else:
                gloss = base_name

        return gloss

    def determine_handedness(self, frames: List[Dict]) -> str:
        """Determine handedness based on first and last 25% of frames"""
        if not frames:
            return "unknown"

        num_frames = len(frames)
        quarter_size = max(1, num_frames // 4)
        important_frames = frames[:quarter_size] + frames[-quarter_size:]

        left_count = 0
        right_count = 0

        for frame in important_frames:
            hands_present = set()

            if "hands" in frame and frame["hands"]:
                for hand in frame["hands"]:
                    hands_present.add(hand["handedness"])

            # Fixed: Move these checks outside the if/elif chain
            if "Left" in hands_present:
                left_count += 1
            if "Right" in hands_present:
                right_count += 1

        total_frames = len(important_frames)
        left_ratio = left_count / total_frames if total_frames > 0 else 0
        right_ratio = right_count / total_frames if total_frames > 0 else 0

        if left_ratio > right_ratio:
            return "left"
        elif right_ratio > left_ratio:
            return "right"
        else:
            return "unknown"

    def calculate_hand_consistency(self, frames: List[Dict]) -> Dict[str, float]:
        """Calculate hand consistency metrics to detect flickering"""
        if not frames:
            return {
                "left_consistency": 0.0,
                "right_consistency": 0.0,
                "overall_penalty": 1.0,
            }

        left_present = []
        right_present = []

        for frame in frames:
            left_found = False
            right_found = False

            if "hands" in frame and frame["hands"]:
                for hand in frame["hands"]:
                    if hand["handedness"] == "Left":
                        left_found = True
                    elif hand["handedness"] == "Right":
                        right_found = True

            left_present.append(left_found)
            right_present.append(right_found)

        def calculate_consistency_score(presence_list: List[bool]) -> float:
            if len(presence_list) <= 1:
                return 1.0

            transitions = sum(
                1
                for i in range(1, len(presence_list))
                if presence_list[i] != presence_list[i - 1]
            )
            max_transitions = len(presence_list) - 1

            if max_transitions == 0:
                return 1.0

            transition_ratio = transitions / max_transitions
            return max(0.0, 1.0 - transition_ratio)

        def calculate_single_frame_penalty(presence_list: List[bool]) -> float:
            if len(presence_list) <= 2:
                return 1.0

            single_frame_count = 0
            total_appearances = 0

            i = 0
            while i < len(presence_list):
                if presence_list[i]:
                    consecutive = 1
                    j = i + 1
                    while j < len(presence_list) and presence_list[j]:
                        consecutive += 1
                        j += 1

                    total_appearances += 1
                    if consecutive == 1:
                        single_frame_count += 1

                    i = j
                else:
                    i += 1

            if total_appearances == 0:
                return 1.0

            single_frame_ratio = single_frame_count / total_appearances
            return max(0.1, 1.0 - single_frame_ratio)

        left_consistency = calculate_consistency_score(left_present)
        right_consistency = calculate_consistency_score(right_present)
        left_single_penalty = calculate_single_frame_penalty(left_present)
        right_single_penalty = calculate_single_frame_penalty(right_present)

        left_weight = sum(left_present) / len(left_present) if left_present else 0
        right_weight = sum(right_present) / len(right_present) if right_present else 0

        if left_weight == 0 and right_weight == 0:
            overall_penalty = 0.1
        elif left_weight == 0:
            overall_penalty = right_consistency * right_single_penalty
        elif right_weight == 0:
            overall_penalty = left_consistency * left_single_penalty
        else:
            total_weight = left_weight + right_weight
            overall_penalty = (
                left_weight / total_weight
            ) * left_consistency * left_single_penalty + (
                right_weight / total_weight
            ) * right_consistency * right_single_penalty

        return {
            "left_consistency": left_consistency,
            "right_consistency": right_consistency,
            "left_single_penalty": left_single_penalty,
            "right_single_penalty": right_single_penalty,
            "left_presence_ratio": left_weight,
            "right_presence_ratio": right_weight,
            "overall_penalty": overall_penalty,
        }

    def get_gesture_files(
        self, handedness_filter: Optional[str] = None
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Get all gesture files organized by gloss with handedness"""
        gesture_files = defaultdict(list)

        for root, dirs, files in os.walk(LANDMARKS_DIR_HANDS_ONLY):
            for file in files:
                if file.endswith("_landmarks.pkl"):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, "rb") as f:
                            data = pickle.load(f)

                        handedness = self.determine_handedness(data["frames"])

                        if handedness_filter and handedness != handedness_filter:
                            continue

                        gloss = self.extract_gloss_from_filename(file)
                        gesture_files[gloss].append((file_path, handedness))

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue

        return gesture_files

    def select_representative_file(
        self, files: List[Tuple[str, str]], gloss: str
    ) -> Tuple[str, str, Dict]:
        """Select the most representative file for a gloss"""
        if len(files) == 1:
            file_path, handedness = files[0]
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            return (
                file_path,
                handedness,
                {
                    "frames": len(data["frames"]),
                    "hand_coverage": 1.0,
                    "consistency_penalty": 1.0,
                },
            )

        file_stats = []
        for file_path, handedness in files:
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)

                frames = data["frames"]
                num_frames = len(frames)

                quarter_size = max(1, num_frames // 4)
                important_frames = frames[:quarter_size] + frames[-quarter_size:]

                hand_frames = sum(1 for frame in important_frames if frame.get("hands"))
                hand_coverage = (
                    hand_frames / len(important_frames) if important_frames else 0
                )

                consistency_metrics = self.calculate_hand_consistency(frames)
                consistency_penalty = consistency_metrics["overall_penalty"]

                length_score = 1.0
                if num_frames < 10:
                    length_score = 0.5
                elif num_frames > 40:
                    length_score = 0.8

                base_score = hand_coverage * length_score * min(num_frames, 30)
                total_score = base_score * consistency_penalty

                file_stats.append(
                    {
                        "path": file_path,
                        "handedness": handedness,
                        "frames": num_frames,
                        "hand_coverage": hand_coverage,
                        "consistency_penalty": consistency_penalty,
                        "score": total_score,
                    }
                )

            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                continue

        if not file_stats:
            return files[0][0], files[0][1], {}

        best_file = max(file_stats, key=lambda x: x["score"])

        print(
            f"Gloss '{gloss}' ({best_file['handedness']}): {len(files)} files, "
            f"selected {os.path.basename(best_file['path'])}"
        )
        print(
            f"  Frames: {best_file['frames']}, Coverage: {best_file['hand_coverage']:.2f}, "
            f"Consistency: {best_file['consistency_penalty']:.2f}"
        )

        if best_file["consistency_penalty"] < 0.7:
            print(
                f"  WARNING: Selected file has poor hand consistency (flickering detected)"
            )

        return (
            best_file["path"],
            best_file["handedness"],
            {
                "frames": best_file["frames"],
                "hand_coverage": best_file["hand_coverage"],
                "consistency_penalty": best_file["consistency_penalty"],
            },
        )

    def create_metadata(
        self, handedness_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create metadata file with representative gestures"""
        print("=== Creating Gesture Representatives Metadata ===")

        gesture_files = self.get_gesture_files(handedness_filter)
        print(f"Found {len(gesture_files)} unique gestures")

        if handedness_filter:
            print(f"Filtering for {handedness_filter} handedness only")

        representatives = {}
        handedness_counts = defaultdict(int)

        for gloss, files in gesture_files.items():
            if files:
                rep_file, handedness, stats = self.select_representative_file(
                    files, gloss
                )
                representatives[gloss] = {
                    "file_path": rep_file,
                    "handedness": handedness,
                    "stats": stats,
                    "alternatives": [f for f, h in files if f != rep_file],
                }
                handedness_counts[handedness] += 1

        print(f"\nSelected {len(representatives)} representative gestures:")
        for handedness, count in handedness_counts.items():
            print(f"  {handedness}: {count} gestures")

        metadata = {
            "created_at": datetime.now().isoformat(),
            "handedness_filter": handedness_filter,
            "total_glosses": len(representatives),
            "handedness_distribution": dict(handedness_counts),
            "representatives": representatives,
        }

        output_dir = Path(OUTPUT_DIR) / "gesture_metadata"
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_file = (
            output_dir
            / f"representatives{'_' + handedness_filter if handedness_filter else ''}.json"
        )
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n=== Metadata Creation Complete ===")
        print(f"Saved to: {metadata_file}")

        return metadata


class GestureTransitionGenerator:
    """Generates transitions on-demand from representative gestures"""

    def __init__(self, metadata_path: str):
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        self.representatives = self.metadata["representatives"]

    def select_random_glosses(self, count: int = 20) -> List[str]:
        """Select random glosses from available representatives"""
        available_glosses = list(self.representatives.keys())

        if len(available_glosses) < count:
            print(
                f"Warning: Only {len(available_glosses)} glosses available, using all of them"
            )
            return available_glosses

        selected = random.sample(available_glosses, count)
        print(
            f"Randomly selected {len(selected)} glosses from {len(available_glosses)} available"
        )
        return selected

    def extract_hand_positions(self, frames: List[Dict]) -> np.ndarray:
        """Extract hand positions from frames"""
        positions = []
        for frame in frames:
            frame_positions = np.zeros((2, 21, 3))
            if "hands" in frame and frame["hands"]:
                for hand in frame["hands"]:
                    hand_idx = 0 if hand["handedness"] == "Left" else 1
                    landmarks = np.array(hand["landmarks"])
                    if landmarks.shape == (21, 3):
                        frame_positions[hand_idx] = landmarks
            positions.append(frame_positions)
        return np.array(positions)

    def extract_pose_positions(self, frames: List[Dict]) -> np.ndarray:
        """Extract pose positions from frames"""
        positions = []
        for frame in frames:
            frame_positions = np.zeros((33, 4))
            if "pose" in frame and frame["pose"] and frame["pose"]["landmarks"]:
                landmarks = np.array(frame["pose"]["landmarks"])
                if landmarks.shape == (33, 4):
                    frame_positions = landmarks
            positions.append(frame_positions)
        return np.array(positions)

    def interpolate_positions(
        self, start_pos: np.ndarray, end_pos: np.ndarray, num_frames: int
    ) -> np.ndarray:
        """Linear interpolation between positions"""
        original_shape = start_pos.shape
        start_flat = start_pos.flatten()
        end_flat = end_pos.flatten()
        weights = np.linspace(0, 1, num_frames)

        interpolated = []
        for weight in weights:
            frame_values = start_flat + weight * (end_flat - start_flat)
            interpolated.append(frame_values)

        interpolated = np.array(interpolated)
        return interpolated.reshape((num_frames,) + original_shape)

    def positions_to_frames(
        self, hand_positions: np.ndarray, pose_positions: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """Convert position arrays back to frame format"""
        frames = []
        num_frames = hand_positions.shape[0]

        for i in range(num_frames):
            frame_data = {"frame_number": i, "hands": []}

            for hand_idx in range(2):
                hand_landmarks = hand_positions[i, hand_idx]
                if np.any(hand_landmarks != 0):
                    hand_data = {
                        "hand_index": hand_idx,
                        "handedness": "Left" if hand_idx == 0 else "Right",
                        "landmarks": hand_landmarks.tolist(),
                    }
                    frame_data["hands"].append(hand_data)

            if pose_positions is not None:
                pose_landmarks = pose_positions[i]
                if np.any(pose_landmarks != 0):
                    frame_data["pose"] = {"landmarks": pose_landmarks.tolist()}
                else:
                    frame_data["pose"] = None

            frames.append(frame_data)

        return frames

    def generate_sequence(
        self,
        gloss_sequence: List[str],
        transition_length: int = 6,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a sequence with transitions from multiple glosses"""
        print(f"=== Generating Sequence ===")
        print(f"Glosses: {' -> '.join(gloss_sequence)}")
        print(f"Transition length: {transition_length} frames")

        # Validate all glosses exist
        denied_glosses = []
        allowed_glosses = []
        for gloss in gloss_sequence:
            if gloss not in self.representatives:
                denied_glosses.append(gloss)
            else:
                allowed_glosses.append(gloss)

        # Load all gesture data
        gesture_data = []
        for gloss in allowed_glosses:
            rep_info = self.representatives[gloss]
            file_path = rep_info["file_path"]

            with open(file_path, "rb") as f:
                data = pickle.load(f)

            gesture_data.append(
                {"gloss": gloss, "data": data, "handedness": rep_info["handedness"]}
            )
            print(
                f"  Loaded {gloss} ({rep_info['handedness']}): {len(data['frames'])} frames"
            )

        # Build combined sequence with transitions
        combined_frames = []
        sequence_info = []

        for i, gesture in enumerate(gesture_data):
            frames = gesture["data"]["frames"]

            if i == 0:
                # First gesture: use all frames
                combined_frames.extend(frames)
                sequence_info.append(
                    {
                        "gloss": gesture["gloss"],
                        "type": "gesture",
                        "frame_range": [0, len(frames) - 1],
                        "original_frames": len(frames),
                    }
                )
            else:
                # Generate transition from previous gesture
                prev_gesture = gesture_data[i - 1]
                prev_frames = prev_gesture["data"]["frames"]

                # Extract positions for transition
                prev_hand_pos = self.extract_hand_positions(prev_frames)
                curr_hand_pos = self.extract_hand_positions(frames)

                # Interpolate
                transition_hand_pos = self.interpolate_positions(
                    prev_hand_pos[-1], curr_hand_pos[0], transition_length
                )

                # Handle pose if available
                transition_pose_pos = None
                if "pose" in prev_gesture["data"].get("landmark_types", []):
                    prev_pose_pos = self.extract_pose_positions(prev_frames)
                    curr_pose_pos = self.extract_pose_positions(frames)
                    transition_pose_pos = self.interpolate_positions(
                        prev_pose_pos[-1], curr_pose_pos[0], transition_length
                    )

                # Convert to frames
                transition_frames = self.positions_to_frames(
                    transition_hand_pos, transition_pose_pos
                )

                # Add transition (excluding first frame to avoid duplicate)
                transition_start = len(combined_frames)
                combined_frames.extend(transition_frames[1:])

                sequence_info.append(
                    {
                        "from_gloss": prev_gesture["gloss"],
                        "to_gloss": gesture["gloss"],
                        "type": "transition",
                        "frame_range": [transition_start, len(combined_frames) - 1],
                        "transition_frames": len(transition_frames) - 1,
                    }
                )

                # Add current gesture (excluding first frame)
                gesture_start = len(combined_frames)
                combined_frames.extend(frames[1:])

                sequence_info.append(
                    {
                        "gloss": gesture["gloss"],
                        "type": "gesture",
                        "frame_range": [gesture_start, len(combined_frames) - 1],
                        "original_frames": len(frames),
                    }
                )

        # Renumber all frames
        for i, frame in enumerate(combined_frames):
            frame["frame_number"] = i

        # Create output data structure
        output_data = {
            "video_path": "generated_sequence",
            "timestamp": datetime.now().isoformat(),
            "frames": combined_frames,
            "landmark_types": gesture_data[0]["data"].get(
                "landmark_types", ["hand_landmarks"]
            ),
            "sequence_metadata": {
                "gloss_sequence": gloss_sequence,
                "transition_length": transition_length,
                "total_frames": len(combined_frames),
                "sequence_info": sequence_info,
                "gesture_count": len(gloss_sequence),
                "transition_count": len(gloss_sequence) - 1,
            },
            "denied_glosses": denied_glosses,
        }

        # Save if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "wb") as f:
                pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"\n=== Sequence Generated ===")
            print(f"Total frames: {len(combined_frames)}")
            print(f"Saved to: {output_file}")

        return output_data


def main():
    parser = argparse.ArgumentParser(description="Gesture transition system")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Create metadata command
    create_parser = subparsers.add_parser(
        "create-metadata", help="Create representatives metadata"
    )
    create_parser.add_argument(
        "--handedness", choices=["left", "right"], help="Filter by handedness"
    )

    # Generate sequence command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate gesture sequence"
    )
    generate_parser.add_argument(
        "--metadata",
        default=REPRESENTATIVES_LEFT,
        help="Path to metadata file",
    )
    generate_parser.add_argument(
        "--glosses",
        nargs="*",
        default=None,
        help="List of glosses to chain (if not provided, 20 random glosses will be selected)",
    )
    generate_parser.add_argument(
        "--random-count",
        type=int,
        default=20,
        help="Number of random glosses to select if --glosses not provided",
    )
    generate_parser.add_argument(
        "--transition-length", type=int, default=4, help="Transition length in frames"
    )
    generate_parser.add_argument(
        "--output",
        default=OUTPUT_DIR / "sythesized_interpolation" / "temp_generated_sequence.pkl",
        help="Output file path",
    )

    args = parser.parse_args()

    if args.command == "create-metadata":
        selector = GestureRepresentativeSelector()
        selector.create_metadata(args.handedness)

    elif args.command == "generate":
        generator = GestureTransitionGenerator(args.metadata)

        # If no glosses provided, select random ones
        if not args.glosses:
            print(
                f"No glosses specified, selecting {args.random_count} random glosses..."
            )
            glosses = generator.select_random_glosses(args.random_count)
        else:
            glosses = args.glosses

        result = generator.generate_sequence(
            glosses, args.transition_length, args.output
        )
        denied_glosses = result["denied_glosses"]

        if denied_glosses:
            print(f"Denied glosses (not found): {', '.join(denied_glosses)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
