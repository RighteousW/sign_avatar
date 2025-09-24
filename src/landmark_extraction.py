import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import argparse
import json
from datetime import datetime

from constants import (
    LANDMARKS_DIR,
    LANDMARKS_DIR_METADATA_JSON,
    LANDMARKS_DIR_METADATA_PKL,
    MEDIAPIPE_HAND_LANDMARKER_PATH,
    MEDIAPIPE_POSE_LANDMARKER_PATH,
    VIDEOS_DIR,
)


class LandmarkExtractor:
    def __init__(self, hand_model_path, pose_model_path, landmark_types):
        """
        Initialize the landmark extractor with MediaPipe models
        """
        self.landmark_types = landmark_types
        self.global_timestamp = 0
        self.processing_metadata = {
            "start_time": datetime.now().isoformat(),
            "landmark_types": landmark_types,
            "model_paths": {
                "hand_model": (
                    hand_model_path if "hand_landmarks" in landmark_types else None
                ),
                "pose_model": (
                    pose_model_path if "pose_landmarks" in landmark_types else None
                ),
            },
            "processed_videos": [],
            "feature_info": None,
            "total_videos": 0,
            "total_frames": 0,
            "failed_videos": [],
        }

        # Initialize hand landmarker if needed
        if "hand_landmarks" in landmark_types:
            hand_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(
                hand_options
            )
        else:
            self.hand_landmarker = None

        # Initialize pose landmarker if needed
        if "pose_landmarks" in landmark_types:
            pose_options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=pose_model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(
                pose_options
            )
        else:
            self.pose_landmarker = None

        # Store feature info in metadata
        self.processing_metadata["feature_info"] = self.get_feature_dimensions()

    def get_feature_dimensions(self):
        """
        Calculate and return the feature dimensions for model building
        """
        hand_landmarks_dim = (
            21 * 3 * 2 if "hand_landmarks" in self.landmark_types else 0
        )  # 126
        pose_landmarks_dim = (
            33 * 4 if "pose_landmarks" in self.landmark_types else 0
        )  # 132

        total_dim = hand_landmarks_dim + pose_landmarks_dim

        return {
            "hand_landmarks": hand_landmarks_dim,
            "pose_landmarks": pose_landmarks_dim,
            "total_features": total_dim,
            "max_hands": 2,
            "hand_landmarks_per_hand": 21 * 3,  # 63
            "pose_total_landmarks": 33,
            "pose_coords_per_landmark": 4,  # x, y, z, visibility
        }

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
            "landmark_types": self.landmark_types,
            "feature_info": self.get_feature_dimensions(),
            "max_feature_vector_size": self.get_feature_dimensions()["total_features"],
        }

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = self.global_timestamp
            self.global_timestamp += 1

            frame_data = {"frame_number": frame_count}

            # Initialize data structures based on selected types
            if "hand_landmarks" in self.landmark_types:
                frame_data["hands"] = []
            if "pose_landmarks" in self.landmark_types:
                frame_data["pose"] = None

            # Hand detection
            if self.hand_landmarker:
                try:
                    hand_result = self.hand_landmarker.detect_for_video(
                        mp_image, timestamp_ms
                    )
                    if hand_result.hand_landmarks:
                        for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                            hand_data = {
                                "hand_index": i,
                                "handedness": (
                                    hand_result.handedness[i][0].category_name
                                    if hand_result.handedness
                                    else "Unknown"
                                ),
                                "landmarks": landmarks,
                            }
                            frame_data["hands"].append(hand_data)
                except Exception as e:
                    print(
                        f"Error extracting hand landmarks from frame {frame_count}: {e}"
                    )

            # Pose detection
            if self.pose_landmarker:
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
                        frame_data["pose"] = {"landmarks": landmarks}
                except Exception as e:
                    print(
                        f"Error extracting pose landmarks from frame {frame_count}: {e}"
                    )

        landmarks_data["frames"].append(frame_data)
        frame_count += 1

        cap.release()

        # Update metadata with video info
        video_metadata = {
            "video_path": video_path,
            "timestamp": timestamp_str,
            "total_frames": frame_count,
            "processing_time": datetime.now().isoformat(),
        }
        self.processing_metadata["processed_videos"].append(video_metadata)
        self.processing_metadata["total_frames"] += frame_count

        return landmarks_data

    def process_video_folder(self, videos_path, landmarks_path):
        """
        Process all videos in the folder structure and save landmarks
        """
        if not os.path.exists(videos_path):
            print(f"Error: Videos folder not found at {videos_path}")
            return

        os.makedirs(landmarks_path, exist_ok=True)

        # Print feature dimensions info
        feature_info = self.get_feature_dimensions()
        print("Feature dimensions:")
        print(f"Hand landmarks: {feature_info['hand_landmarks']}")
        print(f"Pose landmarks: {feature_info['pose_landmarks']}")
        print(f"Total features: {feature_info['total_features']}")
        print(f"Landmark types: {self.landmark_types}")

        video_folders = os.listdir(videos_path)

        for folder_num in video_folders:
            folder_path = os.path.join(videos_path, str(folder_num))
            output_folder_path = os.path.join(landmarks_path, str(folder_num))
            os.makedirs(output_folder_path, exist_ok=True)

            video_files = [f for f in os.listdir(folder_path) if f.endswith(".avi")]

            for video_file in video_files:
                video_path = os.path.join(folder_path, video_file)
                print(f"Processing: {video_path}")

                landmarks_data = self.extract_landmarks_from_video(video_path)

                if landmarks_data is not None:
                    output_filename = video_file.replace(".avi", "_landmarks.pkl")
                    output_file_path = os.path.join(output_folder_path, output_filename)

                    with open(output_file_path, "wb") as f:
                        pickle.dump(landmarks_data, f, protocol=pickle.HIGHEST_PROTOCOL)

                    print(f"Saved landmarks to: {output_file_path}")
                    self.processing_metadata["total_videos"] += 1
                else:
                    print(f"Failed to process: {video_path}")
                    self.processing_metadata["failed_videos"].append(
                        {
                            "video_path": video_path,
                            "error_time": datetime.now().isoformat(),
                        }
                    )

    def save_metadata(self):
        """
        Save processing metadata to both pickle and JSON files
        """
        # Finalize metadata
        self.processing_metadata["end_time"] = datetime.now().isoformat()
        self.processing_metadata["success_rate"] = (
            self.processing_metadata["total_videos"]
            / (
                self.processing_metadata["total_videos"]
                + len(self.processing_metadata["failed_videos"])
            )
            if (
                self.processing_metadata["total_videos"]
                + len(self.processing_metadata["failed_videos"])
            )
            > 0
            else 0
        )

        # Save as pickle
        with open(LANDMARKS_DIR_METADATA_PKL, "wb") as f:
            pickle.dump(self.processing_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save as JSON
        with open(LANDMARKS_DIR_METADATA_JSON, "w") as f:
            json.dump(self.processing_metadata, f, indent=2)

        print(f"Saved metadata to: {LANDMARKS_DIR_METADATA_PKL}")
        print(f"Saved metadata to: {LANDMARKS_DIR_METADATA_JSON}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract landmarks from videos using MediaPipe"
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
        "--output_dir", default=LANDMARKS_DIR, help="Path to save landmarks"
    )
    parser.add_argument(
        "--landmark_types",
        nargs="+",
        choices=["hand_landmarks", "pose_landmarks"],
        default=["hand_landmarks"],
        help="Types of landmarks to extract",
    )

    args = parser.parse_args()

    # Check if required model files exist based on landmark types
    if "hand_landmarks" in args.landmark_types and not os.path.exists(args.hand_model):
        print(f"Error: Hand model file not found: {args.hand_model}")
        return

    if "pose_landmarks" in args.landmark_types and not os.path.exists(args.pose_model):
        print(f"Error: Pose model file not found: {args.pose_model}")
        return

    print("Initializing MediaPipe models...")
    extractor = LandmarkExtractor(args.hand_model, args.pose_model, args.landmark_types)

    print("Starting landmark extraction...")
    extractor.process_video_folder(args.videos_dir, args.output_dir)

    # Save metadata files
    extractor.save_metadata(args.output_dir)

    print("Landmark extraction complete!")


if __name__ == "__main__":
    main()
