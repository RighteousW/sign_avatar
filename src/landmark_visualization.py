import os
import cv2
import pickle
import numpy as np
from tkinter import filedialog
import tkinter as tk
import mediapipe as mp

from constants import ROOT_DIR


class LandmarkVisualizer:
    def __init__(self):
        """Initialize the landmark visualizer using actual metadata from extraction"""
        # Colors for different elements
        self.hand_landmark_color = (0, 255, 0)  # Green
        self.hand_connection_color = (0, 200, 0)  # Dark green
        self.pose_landmark_color = (0, 0, 255)  # Red
        self.pose_connection_color = (0, 100, 255)  # Orange-red

        # Current state
        self.landmarks_data = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.window_name = "Landmark Visualizer"

        # Display settings
        self.frame_width = 640
        self.frame_height = 480

        # Use MediaPipe connections directly
        self.hand_connections = list(mp.solutions.hands.HAND_CONNECTIONS)
        self.pose_connections = list(mp.solutions.pose.POSE_CONNECTIONS)

    def select_files(self):
        """Open file dialog to select landmark files"""
        root = tk.Tk()
        root.withdraw()

        file_paths = filedialog.askopenfilenames(
            title="Select landmark pickle files",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir=ROOT_DIR if os.path.exists(ROOT_DIR) else ".",
        )

        root.destroy()
        return file_paths

    def load_landmarks_data(self, file_paths):
        """Load landmark data from selected files"""
        all_data = []

        for file_path in file_paths:
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)

                all_data.append({"filename": os.path.basename(file_path), "data": data})

                print(f"Loaded: {file_path}")
                print(f"  - Frames: {len(data['frames'])}")
                print(f"  - Timestamp: {data.get('timestamp', 'Unknown')}")

                # Print feature info if available
                if "feature_info" in data:
                    feature_info = data["feature_info"]
                    print(f"  - Total features: {feature_info['total_features']}")
                    print(f"  - Hand landmarks: {feature_info['hand_landmarks']}")
                    print(f"  - Pose landmarks: {feature_info['pose_landmarks']}")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return all_data

    def create_visualization_frame(self, frame_data, filename=""):
        """Create a visualization frame with landmarks and connections"""
        # Create a black background
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        # Add filename and frame info
        cv2.putText(
            frame,
            f"File: {filename}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Frame: {self.current_frame_idx + 1}/{self.total_frames}",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Draw pose landmarks first
        if frame_data.get("pose") and frame_data["pose"]:
            pose_landmarks = frame_data["pose"]["landmarks"]

            # Convert landmarks to pixel coordinates
            pose_points = []
            for landmark in pose_landmarks:
                x = int(landmark[0] * self.frame_width)
                y = int(landmark[1] * self.frame_height)
                pose_points.append((x, y))

            # Draw pose connections
            for connection in self.pose_connections:
                start_idx, end_idx = connection
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

                # Convert landmarks to pixel coordinates
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
                            0.4,
                            self.hand_landmark_color,
                            1,
                        )

        # Add landmarks count info
        hand_count = len(frame_data.get("hands", []))
        pose_available = "Yes" if frame_data.get("pose") else "No"

        cv2.putText(
            frame,
            f"Hands: {hand_count}, Pose: {pose_available}",
            (10, self.frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        return frame

    def show_controls(self):
        """Display control instructions"""
        print("\n" + "=" * 60)
        print("LANDMARK VISUALIZER CONTROLS:")
        print("=" * 60)
        print("n/N     : Next frame")
        print("p/P     : Previous frame")
        print("f/F     : Next file")
        print("b/B     : Previous file")
        print("r/R     : Reset to first frame")
        print("SPACE   : Toggle play/pause")
        print("s/S     : Save current frame")
        print("i/I     : Show file info")
        print("h/H/?   : Show this help")
        print("q/Q/ESC : Quit")
        print("=" * 60)

    def show_file_info(self, all_data, current_file_idx):
        """Display information about loaded files"""
        print("\n" + "=" * 60)
        print("LOADED LANDMARK FILES:")
        print("=" * 60)
        for i, data in enumerate(all_data):
            marker = " >>> " if i == current_file_idx else "     "
            frames_count = len(data["data"]["frames"])
            timestamp = data["data"].get("timestamp", "Unknown")
            print(f"{marker}[{i+1:2d}] {data['filename']}")
            print(f"         Frames: {frames_count}, Timestamp: {timestamp}")

            if "feature_info" in data["data"]:
                feature_info = data["data"]["feature_info"]
                print(f"         Total features: {feature_info['total_features']}")
        print("=" * 60)

    def run(self):
        """Main visualization loop"""
        # Select files
        file_paths = self.select_files()
        if not file_paths:
            print("No files selected. Exiting.")
            return

        # Load landmark data
        all_data = self.load_landmarks_data(file_paths)
        if not all_data:
            print("No valid landmark data found. Exiting.")
            return

        self.show_controls()
        print(f"\nLoaded {len(all_data)} landmark files")

        current_file_idx = 0
        self.current_frame_idx = 0
        playing = False

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        while True:
            # Get current data
            current_data = all_data[current_file_idx]
            self.landmarks_data = current_data["data"]
            filename = current_data["filename"]
            self.total_frames = len(self.landmarks_data["frames"])

            # Ensure frame index is valid
            if self.total_frames > 0:
                self.current_frame_idx = max(
                    0, min(self.current_frame_idx, self.total_frames - 1)
                )
                frame_data = self.landmarks_data["frames"][self.current_frame_idx]

                # Create visualization
                vis_frame = self.create_visualization_frame(frame_data, filename)

                # Add file navigation info
                sequence_info = f"File: {current_file_idx + 1}/{len(all_data)}"
                cv2.putText(
                    vis_frame,
                    sequence_info,
                    (10, self.frame_height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (200, 200, 200),
                    1,
                )

                if playing:
                    cv2.putText(
                        vis_frame,
                        "PLAYING",
                        (self.frame_width - 80, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                cv2.imshow(self.window_name, vis_frame)

            # Handle keyboard input
            key = cv2.waitKey(100 if playing else 0) & 0xFF

            if key in [ord("q"), ord("Q"), 27]:  # Quit
                break
            elif key in [ord("n"), ord("N")]:  # Next frame
                if self.total_frames > 0:
                    self.current_frame_idx = min(
                        self.current_frame_idx + 1, self.total_frames - 1
                    )
            elif key in [ord("p"), ord("P")]:  # Previous frame
                self.current_frame_idx = max(self.current_frame_idx - 1, 0)
            elif key in [ord("f"), ord("F")]:  # Next file
                if current_file_idx < len(all_data) - 1:
                    current_file_idx += 1
                    self.current_frame_idx = 0
                    print(f"Switched to: {all_data[current_file_idx]['filename']}")
            elif key in [ord("b"), ord("B")]:  # Previous file
                if current_file_idx > 0:
                    current_file_idx -= 1
                    self.current_frame_idx = 0
                    print(f"Switched to: {all_data[current_file_idx]['filename']}")
            elif key in [ord("r"), ord("R")]:  # Reset frame
                self.current_frame_idx = 0
            elif key == ord(" "):  # Toggle play/pause
                playing = not playing
                status = "Playing" if playing else "Paused"
                print(f"{status} - {all_data[current_file_idx]['filename']}")
            elif key in [ord("i"), ord("I")]:  # Show file info
                self.show_file_info(all_data, current_file_idx)
            elif key in [ord("h"), ord("H"), ord("?")]:  # Show help
                self.show_controls()
            elif key in [ord("s"), ord("S")]:  # Save frame
                if self.total_frames > 0:
                    save_filename = (
                        f"frame_{current_file_idx}_{self.current_frame_idx}.png"
                    )
                    cv2.imwrite(save_filename, vis_frame)
                    print(f"Saved: {save_filename}")

            # Auto-advance when playing
            if playing and self.total_frames > 0:
                self.current_frame_idx += 1
                if self.current_frame_idx >= self.total_frames:
                    if current_file_idx < len(all_data) - 1:
                        current_file_idx += 1
                        self.current_frame_idx = 0
                        print(
                            f"Auto-switched to: {all_data[current_file_idx]['filename']}"
                        )
                    else:
                        playing = False
                        print("End of all files. Stopped playing.")

        cv2.destroyAllWindows()


def main():
    print("Starting Landmark Visualizer...")
    visualizer = LandmarkVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()
