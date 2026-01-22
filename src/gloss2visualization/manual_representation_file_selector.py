import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
import json
from collections import defaultdict
import re
from datetime import datetime
import cv2
import mediapipe as mp

from ..constants import (
    LANDMARKS_DIR_HANDS_POSE,
    REPRESENTATIVES_MANUAL,
)


class ManualLandmarkVisualizer:
    """Interactive visualizer for manual representative selection"""

    def __init__(self):
        # Colors for different elements
        self.hand_landmark_color = (0, 255, 0)  # Green
        self.hand_connection_color = (0, 200, 0)  # Dark green
        self.pose_landmark_color = (0, 0, 255)  # Red
        self.pose_connection_color = (0, 100, 255)  # Orange-red

        # Current state
        self.landmarks_data = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.window_name = "Representative Selection"

        # Autoplay state
        self.is_autoplaying = False
        self.autoplay_fps = 30  # Default FPS for autoplay
        self.autoplay_delay = int(1000 / self.autoplay_fps)  # milliseconds

        # Display settings
        self.frame_width = 640
        self.frame_height = 480

        # Use MediaPipe connections
        self.hand_connections = list(mp.solutions.hands.HAND_CONNECTIONS)
        self.pose_connections = list(mp.solutions.pose.POSE_CONNECTIONS)

    def create_visualization_frame(
        self, frame_data, filename="", gloss="", file_idx=0, total_files=0
    ):
        """Create a visualization frame with landmarks and connections"""
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        # Add gloss and file info
        cv2.putText(
            frame,
            f"Gloss: {gloss}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"File {file_idx + 1}/{total_files}: {filename}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Add autoplay indicator
        autoplay_text = "PLAYING" if self.is_autoplaying else "PAUSED"
        autoplay_color = (0, 255, 0) if self.is_autoplaying else (0, 0, 255)
        cv2.putText(
            frame,
            f"Frame: {self.current_frame_idx + 1}/{self.total_frames} [{autoplay_text}]",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            autoplay_color,
            1,
        )

        # Draw pose landmarks first (if available)
        if frame_data.get("pose") and frame_data["pose"]:
            pose_landmarks = frame_data["pose"]["landmarks"]
            pose_points = []

            for landmark in pose_landmarks:
                x = int(landmark[0] * self.frame_width)
                y = int(landmark[1] * self.frame_height)
                pose_points.append((x, y))

            # Draw pose connections
            for connection in self.pose_connections:
                start_idx, end_idx = connection
                # Adjust indices for filtered landmarks (removed 17-22)
                if start_idx >= 17:
                    start_idx -= 6
                if end_idx >= 17:
                    end_idx -= 6

                if (
                    start_idx < len(pose_points)
                    and end_idx < len(pose_points)
                    and 0 <= pose_points[start_idx][0] < self.frame_width
                    and 0 <= pose_points[start_idx][1] < self.frame_height
                    and 0 <= pose_points[end_idx][0] < self.frame_width
                    and 0 <= pose_points[end_idx][1] < self.frame_height
                ):
                    cv2.line(
                        frame,
                        pose_points[start_idx],
                        pose_points[end_idx],
                        self.pose_connection_color,
                        2,
                    )

            # Draw pose landmarks
            for i, (x, y) in enumerate(pose_points):
                if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                    cv2.circle(frame, (x, y), 3, self.pose_landmark_color, -1)

        # Draw hand landmarks
        if frame_data.get("hands"):
            for hand_idx, hand_data in enumerate(frame_data["hands"]):
                hand_landmarks = hand_data["landmarks"]
                handedness = hand_data.get("handedness", f"Hand_{hand_idx}")

                hand_points = []
                for landmark in hand_landmarks:
                    x = int(landmark[0] * self.frame_width)
                    y = int(landmark[1] * self.frame_height)
                    hand_points.append((x, y))

                # Draw hand connections
                for connection in self.hand_connections:
                    start_idx, end_idx = connection
                    if (
                        start_idx < len(hand_points)
                        and end_idx < len(hand_points)
                        and 0 <= hand_points[start_idx][0] < self.frame_width
                        and 0 <= hand_points[start_idx][1] < self.frame_height
                        and 0 <= hand_points[end_idx][0] < self.frame_width
                        and 0 <= hand_points[end_idx][1] < self.frame_height
                    ):
                        cv2.line(
                            frame,
                            hand_points[start_idx],
                            hand_points[end_idx],
                            self.hand_connection_color,
                            2,
                        )

                # Draw hand landmarks
                for i, (x, y) in enumerate(hand_points):
                    if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                        cv2.circle(frame, (x, y), 4, self.hand_landmark_color, -1)

                # Add hand label
                if hand_points and len(hand_points) > 0:
                    wrist_x, wrist_y = hand_points[0]
                    if (
                        0 <= wrist_x < self.frame_width
                        and 10 <= wrist_y < self.frame_height
                    ):
                        cv2.putText(
                            frame,
                            handedness,
                            (wrist_x, wrist_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            self.hand_landmark_color,
                            2,
                        )

        # Add controls info at bottom
        controls = "SPACE:Play/Pause  n/p:Frame  N/P:File  A:SELECT  q:Quit"
        cv2.putText(
            frame,
            controls,
            (10, self.frame_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 255),
            1,
        )

        return frame

    def visualize_and_select(self, file_paths: List[str], gloss: str) -> Optional[str]:
        """
        Show files for a gloss and allow user to select one
        Returns: selected file path or None if skipped
        """
        if not file_paths:
            return None

        print(f"\n{'='*60}")
        print(f"Selecting representative for gloss: {gloss}")
        print(f"Total files: {len(file_paths)}")
        print(f"{'='*60}")
        print("CONTROLS:")
        print("  SPACE : Play/Pause autoplay")
        print("  p : Next frame (stops autoplay)")
        print("  o : Previous frame (stops autoplay)")
        print("  m : Next file")
        print("  n : Previous file")
        print("  i : SELECT this file as representative")
        print("  q : Quit (no selection)")
        print(f"{'='*60}\n")

        current_file_idx = 0
        self.current_frame_idx = 0
        self.is_autoplaying = False

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        while True:
            # Load current file
            current_file = file_paths[current_file_idx]

            try:
                with open(current_file, "rb") as f:
                    self.landmarks_data = pickle.load(f)

                self.total_frames = len(self.landmarks_data["frames"])

                # Ensure frame index is valid
                if self.total_frames > 0:
                    self.current_frame_idx = max(
                        0, min(self.current_frame_idx, self.total_frames - 1)
                    )
                    frame_data = self.landmarks_data["frames"][self.current_frame_idx]

                    filename = os.path.basename(current_file)
                    vis_frame = self.create_visualization_frame(
                        frame_data, filename, gloss, current_file_idx, len(file_paths)
                    )

                    cv2.imshow(self.window_name, vis_frame)

            except Exception as e:
                print(f"Error loading {current_file}: {e}")
                # Try next file
                if current_file_idx < len(file_paths) - 1:
                    current_file_idx += 1
                    continue
                else:
                    break

            # Handle keyboard input with appropriate delay
            wait_time = self.autoplay_delay if self.is_autoplaying else 0
            key = cv2.waitKey(wait_time) & 0xFF

            # Handle autoplay
            if self.is_autoplaying and key == 255:  # No key pressed
                if self.total_frames > 0:
                    self.current_frame_idx += 1
                    if self.current_frame_idx >= self.total_frames:
                        self.current_frame_idx = 0  # Loop back to start
                continue

            if key in [ord("q"), ord("Q"), 27]:  # Quit
                cv2.destroyWindow(self.window_name)
                return None

            elif key == ord("i"):  # SELECT
                print(f"✓ Selected: {os.path.basename(current_file)}")
                cv2.destroyWindow(self.window_name)
                return current_file

            elif key == ord(" "):  # SPACE - Toggle autoplay
                self.is_autoplaying = not self.is_autoplaying
                status = "STARTED" if self.is_autoplaying else "STOPPED"
                print(f"Autoplay {status}")

            elif key == ord("p"):  # Next frame (stops autoplay)
                self.is_autoplaying = False
                if self.total_frames > 0:
                    self.current_frame_idx = min(
                        self.current_frame_idx + 1, self.total_frames - 1
                    )

            elif key == ord("o"):  # Previous frame (stops autoplay)
                self.is_autoplaying = False
                self.current_frame_idx = max(self.current_frame_idx - 1, 0)

            elif key == ord("m"):  # Next file
                self.is_autoplaying = False
                if current_file_idx < len(file_paths) - 1:
                    current_file_idx += 1
                    self.current_frame_idx = 0
                    print(f"→ File {current_file_idx + 1}/{len(file_paths)}")
                else:
                    print("Already at last file")

            elif key == ord("n"):  # Previous file
                self.is_autoplaying = False
                if current_file_idx > 0:
                    current_file_idx -= 1
                    self.current_frame_idx = 0
                    print(f"← File {current_file_idx + 1}/{len(file_paths)}")
                else:
                    print("Already at first file")

        cv2.destroyWindow(self.window_name)
        return None


class ManualGestureRepresentativeSelector:
    """Manual selection of representative gesture files with visualization"""

    def __init__(self, landmarks_dir: str):
        self.landmarks_dir = landmarks_dir
        self.visualizer = ManualLandmarkVisualizer()

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

    def get_files_by_gloss(self) -> Dict[str, List[str]]:
        """Get all files organized by gloss"""
        files_by_gloss = defaultdict(list)

        for root, dirs, files in os.walk(self.landmarks_dir):
            for file in files:
                if file.endswith("_landmarks.pkl"):
                    file_path = os.path.join(root, file)
                    gloss = self.extract_gloss_from_filename(file)
                    files_by_gloss[gloss].append(file_path)

        # Sort files within each gloss
        for gloss in files_by_gloss:
            files_by_gloss[gloss].sort()

        return dict(files_by_gloss)

    def create_metadata(
        self
    ) -> Dict[str, Any]:
        """Create metadata file with manually selected representatives"""
        print("=== Manual Representative Selection ===")
        print(f"Landmarks directory: {self.landmarks_dir}")

        files_by_gloss = self.get_files_by_gloss()
        print(f"Found {len(files_by_gloss)} unique glosses")

        representatives = {}
        skipped_glosses = []

        for gloss_idx, (gloss, file_paths) in enumerate(sorted(files_by_gloss.items())):
            print(
                f"\n[{gloss_idx + 1}/{len(files_by_gloss)}] Processing gloss: {gloss}"
            )

            if len(file_paths) == 1:
                print(
                    f"Only one file available, auto-selecting: {os.path.basename(file_paths[0])}"
                )
                selected_file = file_paths[0]
            else:
                selected_file = self.visualizer.visualize_and_select(file_paths, gloss)

            if selected_file:
                # Load selected file to get handedness info
                try:
                    with open(selected_file, "rb") as f:
                        data = pickle.load(f)

                    # Determine handedness from first few frames
                    handedness = self.determine_handedness(data["frames"])

                    representatives[gloss] = {
                        "file_path": str(selected_file),
                        "handedness": handedness,
                        "stats": {
                            "frames": len(data["frames"]),
                            "manual_selection": True,
                        },
                        "alternatives": [str(f) for f in file_paths if f != selected_file],
                    }

                    # Save immediately after each selection
                    temp_metadata = {
                        "created_at": datetime.now().isoformat(),
                        "selection_method": "manual",
                        "landmarks_dir": self.landmarks_dir,
                        "total_glosses": len(representatives),
                        "skipped_glosses": skipped_glosses,
                        "representatives": representatives,
                    }
                    with open(REPRESENTATIVES_MANUAL, "w") as f:
                        json.dump(temp_metadata, f, indent=2)
                    print(f"  ✓ Saved progress ({len(representatives)} glosses)")

                except Exception as e:
                    print(f"Error processing selected file: {e}")
                    skipped_glosses.append(gloss)
            else:
                print(f"✗ Skipped gloss: {gloss}")
                skipped_glosses.append(gloss)

                # Save skip immediately too
                temp_metadata = {
                    "created_at": datetime.now().isoformat(),
                    "selection_method": "manual",
                    "landmarks_dir": self.landmarks_dir,
                    "total_glosses": len(representatives),
                    "skipped_glosses": skipped_glosses,
                    "representatives": representatives,
                }
                with open(REPRESENTATIVES_MANUAL, "w") as f:
                    json.dump(temp_metadata, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✓ Selected {len(representatives)} representatives")
        if skipped_glosses:
            print(
                f"✗ Skipped {len(skipped_glosses)} glosses: {', '.join(skipped_glosses[:5])}"
            )
            if len(skipped_glosses) > 5:
                print(f"  ... and {len(skipped_glosses) - 5} more")

        # Create metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "selection_method": "manual",
            "landmarks_dir": self.landmarks_dir,
            "total_glosses": len(representatives),
            "skipped_glosses": skipped_glosses,
            "representatives": representatives,
        }

        # Save metadata
        with open(REPRESENTATIVES_MANUAL, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n=== Selection Complete ===")
        print(f"Saved to: {REPRESENTATIVES_MANUAL}")

        return metadata

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
            if "hands" in frame and frame["hands"]:
                for hand in frame["hands"]:
                    if hand["handedness"] == "Left":
                        left_count += 1
                    elif hand["handedness"] == "Right":
                        right_count += 1

        if left_count > right_count:
            return "left"
        elif right_count > left_count:
            return "right"
        else:
            return "unknown"


def main():
    print(f"Using landmarks from: {LANDMARKS_DIR_HANDS_POSE}")

    # Create selector and run
    selector = ManualGestureRepresentativeSelector(str(LANDMARKS_DIR_HANDS_POSE))
    selector.create_metadata()


if __name__ == "__main__":
    main()
