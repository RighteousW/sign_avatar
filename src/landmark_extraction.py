import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import argparse
import numpy as np

from constants import (
    LANDMARKS_DIR,
    MEDIAPIPE_HAND_LANDMARKER_PATH,
    MEDIAPIPE_POSE_LANDMARKER_PATH,
    VIDEOS_DIR,
)

# MediaPipe connections
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS


class LandmarkExtractor:
    def __init__(self, hand_model_path, pose_model_path):
        """
        Initialize the landmark extractor with MediaPipe models
        """
        # Initialize hand landmarker for VIDEO mode
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        # Initialize pose landmarker for VIDEO mode
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        # Global timestamp counter that increases across all videos
        self.global_timestamp = 0

        # Store connection information
        self.hand_connections = list(HAND_CONNECTIONS)
        self.pose_connections = list(POSE_CONNECTIONS)

    def get_feature_dimensions(self):
        """
        Calculate and return the feature dimensions for model building
        """
        # Hand landmarks: 21 points * 3 coordinates (x, y, z) per hand * max 2 hands
        hand_landmarks_dim = 21 * 3 * 2  # 126

        # Hand connections: 20 connections per hand * max 2 hands
        hand_connections_dim = len(self.hand_connections) * 2  # 40

        # Pose landmarks: 33 points * 4 coordinates (x, y, z, visibility)
        pose_landmarks_dim = 33 * 4  # 132

        # Pose connections: 35 connections
        pose_connections_dim = len(self.pose_connections)  # 35

        total_dim = (
            hand_landmarks_dim
            + hand_connections_dim
            + pose_landmarks_dim
            + pose_connections_dim
        )

        feature_info = {
            "hand_landmarks": hand_landmarks_dim,
            "hand_connections": hand_connections_dim,
            "pose_landmarks": pose_landmarks_dim,
            "pose_connections": pose_connections_dim,
            "total_features": total_dim,
            "breakdown": {
                "hand_landmarks_per_hand": 21 * 3,  # 63
                "max_hands": 2,
                "pose_landmarks_total": 33 * 4,  # 132
                "hand_connections_per_hand": len(self.hand_connections),  # 20
                "pose_connections_total": len(self.pose_connections),  # 35
            },
        }

        return feature_info

    def calculate_connection_features(self, landmarks, connections):
        """
        Calculate connection features (distances between connected landmarks)
        """
        if not landmarks:
            return []

        connection_features = []
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = np.array(landmarks[start_idx][:3])  # x, y, z
                end_point = np.array(landmarks[end_idx][:3])
                distance = np.linalg.norm(end_point - start_point)
                connection_features.append(distance)
            else:
                connection_features.append(0.0)  # Missing landmark

        return connection_features

    def extract_landmarks_from_video(self, video_path):
        """
        Extract landmarks from video using VIDEO mode
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        filename = os.path.basename(video_path)
        timestamp_str = filename.split("_", 1)[1].replace(".avi", "")

        landmarks_data = {
            "video_path": video_path,
            "timestamp": timestamp_str,
            "frames": [],
            "connections": {
                "hand_connections": self.hand_connections,
                "pose_connections": self.pose_connections,
            },
            "feature_info": self.get_feature_dimensions(),
        }

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Use global timestamp that increases across all videos
            timestamp_ms = self.global_timestamp
            self.global_timestamp += 1

            frame_data = {
                "frame_number": frame_count,
                "hands": [],
                "pose": None,
                "connection_features": {"hands": [], "pose": []},
            }

            # Hand detection
            try:
                hand_result = self.hand_landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )
                if hand_result.hand_landmarks:
                    for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]

                        # Calculate hand connection features
                        hand_connections = self.calculate_connection_features(
                            landmarks, self.hand_connections
                        )

                        hand_data = {
                            "hand_index": i,
                            "handedness": (
                                hand_result.handedness[i][0].category_name
                                if hand_result.handedness
                                else "Unknown"
                            ),
                            "landmarks": landmarks,
                            "connection_features": hand_connections,
                        }
                        frame_data["hands"].append(hand_data)
                        frame_data["connection_features"]["hands"].append(
                            hand_connections
                        )
            except Exception as e:
                print(f"Error extracting hand landmarks from frame {frame_count}: {e}")

            # Pose detection
            try:
                pose_result = self.pose_landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )
                if pose_result.pose_landmarks:
                    pose_landmarks = pose_result.pose_landmarks[0]
                    landmarks = []
                    for lm in pose_landmarks:
                        landmarks.append(
                            [lm.x, lm.y, lm.z, getattr(lm, "visibility", None)]
                        )

                    # Calculate pose connection features
                    pose_connections = self.calculate_connection_features(
                        landmarks, self.pose_connections
                    )

                    frame_data["pose"] = {
                        "landmarks": landmarks,
                        "connection_features": pose_connections,
                    }
                    frame_data["connection_features"]["pose"] = pose_connections
            except Exception as e:
                print(f"Error extracting pose landmarks from frame {frame_count}: {e}")

            landmarks_data["frames"].append(frame_data)
            frame_count += 1

        cap.release()
        return landmarks_data

    def process_video_folder(self, videos_path, landmarks_path):
        """
        Process all videos in the folder structure and save landmarks
        """
        if not os.path.exists(videos_path):
            print(f"Error: Videos folder not found at {videos_path}")
            return

        os.makedirs(landmarks_path, exist_ok=True)

        # Print feature dimensions info once
        feature_info = self.get_feature_dimensions()
        print("\n" + "=" * 60)
        print("FEATURE DIMENSIONS FOR MODEL BUILDING")
        print("=" * 60)
        print(f"Hand landmarks (max 2 hands): {feature_info['hand_landmarks']}")
        print(f"Hand connections (max 2 hands): {feature_info['hand_connections']}")
        print(f"Pose landmarks: {feature_info['pose_landmarks']}")
        print(f"Pose connections: {feature_info['pose_connections']}")
        print(f"TOTAL FEATURES: {feature_info['total_features']}")
        print("\nBreakdown:")
        print(
            f"  - Hand landmarks per hand: {feature_info['breakdown']['hand_landmarks_per_hand']}"
        )
        print(
            f"  - Hand connections per hand: {feature_info['breakdown']['hand_connections_per_hand']}"
        )
        print(
            f"  - Pose landmarks total: {feature_info['breakdown']['pose_landmarks_total']}"
        )
        print(
            f"  - Pose connections total: {feature_info['breakdown']['pose_connections_total']}"
        )
        print("=" * 60 + "\n")

        # Get all folders
        video_folders = os.listdir(videos_path)

        for folder_num in video_folders:
            folder_path = os.path.join(videos_path, str(folder_num))
            output_folder_path = os.path.join(landmarks_path, str(folder_num))
            os.makedirs(output_folder_path, exist_ok=True)

            # Get all .avi files in the folder
            video_files = [f for f in os.listdir(folder_path) if f.endswith(".avi")]

            for video_file in video_files:
                video_path = os.path.join(folder_path, video_file)

                print(f"Processing: {video_path}")
                landmarks_data = self.extract_landmarks_from_video(video_path)

                if landmarks_data is not None:
                    # Save landmarks as pickle file
                    output_filename = video_file.replace(".avi", "_landmarks.pkl")
                    output_file_path = os.path.join(output_folder_path, output_filename)

                    with open(output_file_path, "wb") as f:
                        pickle.dump(landmarks_data, f, protocol=pickle.HIGHEST_PROTOCOL)

                    print(f"Saved landmarks to: {output_file_path}")
                else:
                    print(f"Failed to process: {video_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract landmarks and connections from videos using MediaPipe"
    )
    parser.add_argument(
        "--hand_model",
        default=MEDIAPIPE_HAND_LANDMARKER_PATH,
        help="Path to MediaPipe hand landmark .task model",
    )
    parser.add_argument(
        "--pose_model",
        default=MEDIAPIPE_POSE_LANDMARKER_PATH,
        help="Path to MediaPipe pose landmark .task model",
    )
    parser.add_argument(
        "--videos_dir", default=VIDEOS_DIR, help="Path to folder containing videos"
    )
    parser.add_argument(
        "--output_dir",
        default=LANDMARKS_DIR,
        help="Path to save landmarks",
    )

    args = parser.parse_args()

    # Check if model files exist
    if not os.path.exists(args.hand_model):
        print(f"Error: Hand model file not found: {args.hand_model}")
        return

    if not os.path.exists(args.pose_model):
        print(f"Error: Pose model file not found: {args.pose_model}")
        return

    # Initialize extractor
    print("Initializing MediaPipe models...")
    extractor = LandmarkExtractor(args.hand_model, args.pose_model)

    # Process videos
    print("Starting landmark extraction...")
    extractor.process_video_folder(args.videos_dir, args.output_dir)

    print("Landmark extraction complete!")


if __name__ == "__main__":
    main()
