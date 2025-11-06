import cv2
import torch
import pickle
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import json
from pathlib import Path
import argparse

from ..constants import (
    MEDIAPIPE_HAND_LANDMARKER_PATH,
    MEDIAPIPE_POSE_LANDMARKER_PATH,
    get_gesture_metadata_path,
    get_gesture_model_path,
)

from ..model_training import GestureRecognizerModel


class PerformanceMeasurementSystem:
    """Measure detailed performance metrics for gesture recognition models"""

    def __init__(
        self,
        hand_model_path,
        pose_model_path,
        use_pose=True,
        skip_rate=0,
    ):
        self.use_pose = use_pose
        self.skip_rate = skip_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n{'='*60}")
        print(
            f"Loading model: {'Hands+Pose' if use_pose else 'Hands-Only'}, {skip_rate}-skip"
        )
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        # Load model metadata
        metadata_path = get_gesture_metadata_path(use_pose, skip_rate)
        with open(str(metadata_path), "rb") as f:
            self.model_info = pickle.load(f)

        # Load model
        self.model = GestureRecognizerModel(
            input_size=self.model_info["input_size"],
            num_classes=len(self.model_info["class_names"]),
            hidden_size=self.model_info["hidden_size"],
            dropout=self.model_info["dropout"],
        )

        checkpoint = torch.load(
            str(get_gesture_model_path(use_pose, skip_rate)),
            map_location=self.device,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print("✓ Model loaded successfully")

        # Setup MediaPipe
        self._setup_mediapipe(hand_model_path, pose_model_path)

        # Timing storage
        self.timing_data = {
            "frame_processing_times": [],
            "mediapipe_hand_times": [],
            "mediapipe_pose_times": [],
            "feature_extraction_times": [],
            "model_inference_times": [],
            "total_sequence_times": [],
        }

        self.sequence_length = self.model_info["sequence_length"]
        self.feature_info = self.model_info["feature_info"]
        self.timestamp_ms = 0

    def _setup_mediapipe(self, hand_model_path, pose_model_path):
        """Setup MediaPipe landmarkers"""
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(hand_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        if self.use_pose:
            pose_options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(pose_model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(
                pose_options
            )

    def process_single_frame(self, frame, timestamp_ms):
        """Process a single frame and measure component times"""
        timings = {}

        # Convert frame for MediaPipe
        frame_start = time.perf_counter()
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        conversion_time = time.perf_counter() - frame_start

        frame_data = {"hands": [], "pose": None}

        # Hand landmark detection
        hand_start = time.perf_counter()
        if self.feature_info["hand_landmarks"] > 0:
            hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            if hand_result and hand_result.hand_landmarks:
                for hand_landmarks in hand_result.hand_landmarks:
                    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                    frame_data["hands"].append({"landmarks": landmarks})
        timings["hand_time"] = time.perf_counter() - hand_start

        # Pose landmark detection
        timings["pose_time"] = 0.0
        if self.use_pose and self.feature_info["pose_landmarks"] > 0:
            pose_start = time.perf_counter()
            pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            if pose_result and pose_result.pose_landmarks:
                pose_landmarks = pose_result.pose_landmarks[0]
                landmarks = [
                    [lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)]
                    for lm in pose_landmarks
                ]
                frame_data["pose"] = {"landmarks": landmarks}
            timings["pose_time"] = time.perf_counter() - pose_start

        # Feature extraction
        feature_start = time.perf_counter()
        features = self._extract_features_from_frame_data(frame_data)
        timings["feature_time"] = time.perf_counter() - feature_start

        timings["total_frame_time"] = (
            conversion_time
            + timings["hand_time"]
            + timings["pose_time"]
            + timings["feature_time"]
        )

        return features, timings

    def _extract_features_from_frame_data(self, frame_data):
        """Extract features from detected landmarks"""
        features = []

        # Hand landmarks
        if self.feature_info["hand_landmarks"] > 0:
            max_hands = self.feature_info["max_hands"]
            hand_dim_per_hand = self.feature_info["hand_landmarks_per_hand"]
            hand_features = np.zeros(self.feature_info["hand_landmarks"])

            if frame_data.get("hands"):
                for i, hand_data in enumerate(frame_data["hands"][:max_hands]):
                    if hand_data and "landmarks" in hand_data:
                        landmarks = np.array(hand_data["landmarks"][:21])[
                            :, :3
                        ].flatten()
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

    def pad_or_truncate_sequence(self, sequence, target_length, feature_size):
        """Pad or truncate sequence to target length"""
        if len(sequence) > target_length:
            indices = np.linspace(0, len(sequence) - 1, target_length, dtype=int)
            return np.array([sequence[i] for i in indices])
        else:
            padded = list(sequence)
            while len(padded) < target_length:
                padded.append(np.zeros(feature_size))
            return np.array(padded)

    def run_inference(self, sequence):
        """Run model inference on sequence"""
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        inference_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        inference_time = time.perf_counter() - inference_start

        predicted_class = self.model_info["class_names"][predicted.item()]
        confidence_value = confidence.item()

        return predicted_class, confidence_value, inference_time

    def process_video(self, video_path):
        """Process entire video and collect timing data"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"\nProcessing video: {video_path.name}")
        print(f"Total frames: {total_frames}, FPS: {fps:.2f}")
        print(f"Skip pattern: Processing 1 out of every {self.skip_rate + 1} frames")

        all_features = []
        frame_idx = 0
        last_processed_features = None

        # Extract features with proper skip pattern
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            should_process = frame_idx % (self.skip_rate + 1) == 0

            if should_process:
                # ACTUALLY CALL MEDIAPIPE
                features, timings = self.process_single_frame(frame, self.timestamp_ms)

                # Store timing data (only for processed frames)
                self.timing_data["frame_processing_times"].append(
                    timings["total_frame_time"]
                )
                self.timing_data["mediapipe_hand_times"].append(timings["hand_time"])
                self.timing_data["mediapipe_pose_times"].append(timings["pose_time"])
                self.timing_data["feature_extraction_times"].append(
                    timings["feature_time"]
                )

                last_processed_features = features
                all_features.append(features)
            else:
                # SKIPPED FRAME - Reuse last processed features (no MediaPipe call)
                if last_processed_features is not None:
                    all_features.append(last_processed_features.copy())
                else:
                    # First frames before any processing
                    all_features.append(
                        np.zeros(self.feature_info["total_features"], dtype=np.float32)
                    )

            self.timestamp_ms += 1
            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames...", end="\r")

        print(f"  Processed {frame_idx}/{total_frames} frames... Done!")
        print(
            f"  Actual MediaPipe calls: {len(self.timing_data['frame_processing_times'])}"
        )

        cap.release()

        # Run inference on sequences
        print("  Running inference on sequences...")
        num_sequences = max(1, len(all_features) - self.sequence_length + 1)

        for i in range(0, len(all_features) - self.sequence_length + 1, 10):
            sequence = all_features[i : i + self.sequence_length]
            sequence = self.pad_or_truncate_sequence(
                sequence,
                self.sequence_length,
                self.feature_info["total_features"],
            )

            _, _, inference_time = self.run_inference(sequence)
            self.timing_data["model_inference_times"].append(inference_time)

        # Calculate total sequence processing time
        # For skipped frames, only the processed frames contribute to computation time
        avg_frame_time = np.mean(self.timing_data["frame_processing_times"])
        avg_inference_time = np.mean(self.timing_data["model_inference_times"])

        # Actual frames that need MediaPipe processing per sequence
        frames_actually_processed = self.sequence_length // (self.skip_rate + 1)

        total_sequence_time = (
            avg_frame_time * frames_actually_processed
        ) + avg_inference_time
        self.timing_data["total_sequence_times"].append(total_sequence_time)

        return {
            "video_name": video_path.name,
            "total_frames": frame_idx,
            "frames_actually_processed": len(
                self.timing_data["frame_processing_times"]
            ),
            "processing_ratio": len(self.timing_data["frame_processing_times"])
            / frame_idx,
            "num_sequences_tested": len(self.timing_data["model_inference_times"]),
        }

    def get_statistics(self):
        """Calculate statistics from collected timing data"""
        stats = {
            "model_config": {
                "type": "Hands+Pose" if self.use_pose else "Hands-Only",
                "skip_rate": self.skip_rate,
                "device": str(self.device),
                "sequence_length": self.sequence_length,
            },
            "frame_processing": {
                "mean_ms": np.mean(self.timing_data["frame_processing_times"]) * 1000,
                "std_ms": np.std(self.timing_data["frame_processing_times"]) * 1000,
                "min_ms": np.min(self.timing_data["frame_processing_times"]) * 1000,
                "max_ms": np.max(self.timing_data["frame_processing_times"]) * 1000,
                "total_samples": len(self.timing_data["frame_processing_times"]),
            },
            "mediapipe_hand": {
                "mean_ms": np.mean(self.timing_data["mediapipe_hand_times"]) * 1000,
                "std_ms": np.std(self.timing_data["mediapipe_hand_times"]) * 1000,
            },
            "mediapipe_pose": (
                {
                    "mean_ms": np.mean(self.timing_data["mediapipe_pose_times"]) * 1000,
                    "std_ms": np.std(self.timing_data["mediapipe_pose_times"]) * 1000,
                }
                if self.use_pose
                else None
            ),
            "feature_extraction": {
                "mean_ms": np.mean(self.timing_data["feature_extraction_times"]) * 1000,
                "std_ms": np.std(self.timing_data["feature_extraction_times"]) * 1000,
            },
            "model_inference": {
                "mean_ms": np.mean(self.timing_data["model_inference_times"]) * 1000,
                "std_ms": np.std(self.timing_data["model_inference_times"]) * 1000,
                "min_ms": np.min(self.timing_data["model_inference_times"]) * 1000,
                "max_ms": np.max(self.timing_data["model_inference_times"]) * 1000,
                "total_samples": len(self.timing_data["model_inference_times"]),
            },
            "total_sequence_processing": {
                "mean_ms": (
                    np.mean(self.timing_data["total_sequence_times"]) * 1000
                    if self.timing_data["total_sequence_times"]
                    else (
                        np.mean(self.timing_data["frame_processing_times"])
                        * self.sequence_length
                        * 1000
                        + np.mean(self.timing_data["model_inference_times"]) * 1000
                    )
                ),
                "description": f"Time to process one complete {self.sequence_length}-frame sequence (feature extraction + inference)",
            },
            "computational_efficiency": {
                "frames_processed_per_skip": f"1 out of {self.skip_rate + 1}",
                "actual_mediapipe_calls_per_sequence": self.sequence_length
                // (self.skip_rate + 1),
                "efficiency_gain_vs_0skip": (
                    f"{(self.skip_rate / (self.skip_rate + 1)) * 100:.1f}%"
                    if self.skip_rate > 0
                    else "N/A (baseline)"
                ),
            },
        }

        return stats


def print_statistics(stats):
    """Print formatted statistics"""
    print(f"\n{'='*70}")
    print(f"PERFORMANCE STATISTICS")
    print(f"{'='*70}")
    print(
        f"\nModel: {stats['model_config']['type']}, {stats['model_config']['skip_rate']}-skip"
    )
    print(f"Device: {stats['model_config']['device']}")
    print(f"Sequence Length: {stats['model_config']['sequence_length']} frames")

    print(f"\n{'─'*70}")
    print("FRAME PROCESSING TIME (single frame)")
    print(f"{'─'*70}")
    fp = stats["frame_processing"]
    print(f"  Mean:    {fp['mean_ms']:.3f} ms")
    print(f"  Std Dev: {fp['std_ms']:.3f} ms")
    print(f"  Min:     {fp['min_ms']:.3f} ms")
    print(f"  Max:     {fp['max_ms']:.3f} ms")
    print(f"  Samples: {fp['total_samples']}")

    print(f"\n{'─'*70}")
    print("COMPONENT BREAKDOWN")
    print(f"{'─'*70}")
    print(
        f"  MediaPipe Hands: {stats['mediapipe_hand']['mean_ms']:.3f} ± {stats['mediapipe_hand']['std_ms']:.3f} ms"
    )
    if stats["mediapipe_pose"]:
        print(
            f"  MediaPipe Pose:  {stats['mediapipe_pose']['mean_ms']:.3f} ± {stats['mediapipe_pose']['std_ms']:.3f} ms"
        )
    print(
        f"  Feature Extract: {stats['feature_extraction']['mean_ms']:.3f} ± {stats['feature_extraction']['std_ms']:.3f} ms"
    )

    print(f"\n{'─'*70}")
    print("MODEL INFERENCE TIME (per sequence)")
    print(f"{'─'*70}")
    mi = stats["model_inference"]
    print(f"  Mean:    {mi['mean_ms']:.3f} ms")
    print(f"  Std Dev: {mi['std_ms']:.3f} ms")
    print(f"  Min:     {mi['min_ms']:.3f} ms")
    print(f"  Max:     {mi['max_ms']:.3f} ms")
    print(f"  Samples: {mi['total_samples']}")

    print(f"\n{'─'*70}")
    print("TOTAL SEQUENCE PROCESSING TIME")
    print(f"{'─'*70}")
    tsp = stats["total_sequence_processing"]
    print(f"  {tsp['description']}")
    print(f"  Mean: {tsp['mean_ms']:.3f} ms")

    print(f"\n{'─'*70}")
    print("COMPUTATIONAL EFFICIENCY")
    print(f"{'─'*70}")
    ce = stats["computational_efficiency"]
    print(f"  Frames processed: {ce['frames_processed_per_skip']}")
    print(
        f"  MediaPipe calls per sequence: {ce['actual_mediapipe_calls_per_sequence']}"
    )
    print(f"  Efficiency gain vs 0-skip: {ce['efficiency_gain_vs_0skip']}")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Measure gesture recognition model performance"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="data/demo_videos",
        help="Directory containing test videos",
    )
    parser.add_argument(
        "--hand_model",
        default=MEDIAPIPE_HAND_LANDMARKER_PATH,
        help="MediaPipe hand model path",
    )
    parser.add_argument(
        "--pose_model",
        default=MEDIAPIPE_POSE_LANDMARKER_PATH,
        help="MediaPipe pose model path",
    )
    parser.add_argument(
        "--use_pose",
        action="store_true",
        default=True,
        help="Use pose landmarks (Hands+Pose model)",
    )
    parser.add_argument(
        "--skip_rate",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Frame skip rate (0=no skip, 1=1-skip, 2=2-skip)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="0-skip_hands_pose_improvement_metrics.json",
        help="Output JSON file for results (optional)",
    )

    args = parser.parse_args()

    # Find videos
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        return

    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    if not video_files:
        print(f"Error: No video files found in {video_dir}")
        return

    print(f"Found {len(video_files)} video(s) to process")

    # Initialize measurement system
    measurement_system = PerformanceMeasurementSystem(
        hand_model_path=args.hand_model,
        pose_model_path=args.pose_model,
        use_pose=args.use_pose,
        skip_rate=args.skip_rate,
    )

    # Process videos
    for video_path in video_files:
        measurement_system.process_video(video_path)

    # Get and print statistics
    stats = measurement_system.get_statistics()
    print_statistics(stats)

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
