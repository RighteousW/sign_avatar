import os
import cv2
import pickle
import numpy as np
from tkinter import filedialog
import tkinter as tk

from constants import LANDMARKS_DIR


class InteractiveLandmarkVisualizer:
    def __init__(self):
        """
        Initialize the interactive landmark visualizer
        """
        # Colors for different elements
        self.hand_landmark_color = (0, 255, 0)  # Green
        self.hand_connection_color = (0, 200, 0)  # Dark green
        self.pose_landmark_color = (0, 0, 255)  # Red
        self.pose_connection_color = (0, 100, 255)  # Orange-red

        # Current state
        self.landmarks_data = None
        self.video_data = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.window_name = "Landmark Visualizer"

        # Create a dummy image for display
        self.frame_width = 640
        self.frame_height = 480

        # Connection data (loaded from pickle files)
        self.hand_connections = []
        self.pose_connections = []

    def select_files(self):
        """
        Open file dialog to select landmark files
        """
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        file_paths = filedialog.askopenfilenames(
            title="Select landmark pickle files",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir=LANDMARKS_DIR if os.path.exists(LANDMARKS_DIR) else ".",
        )

        root.destroy()
        return file_paths

    def load_landmarks_data(self, file_paths):
        """
        Load landmark data from selected files
        """
        all_data = []

        for file_path in file_paths:
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)

                    # Extract connection information from the first file
                    if not self.hand_connections and "connections" in data:
                        self.hand_connections = data["connections"].get(
                            "hand_connections", []
                        )
                        self.pose_connections = data["connections"].get(
                            "pose_connections", []
                        )
                        print(
                            f"Loaded connections: {len(self.hand_connections)} hand, {len(self.pose_connections)} pose"
                        )

                    all_data.append(
                        {"filename": os.path.basename(file_path), "data": data}
                    )
                    print(f"Loaded: {file_path}")
                    print(f"  - Frames: {len(data['frames'])}")
                    print(f"  - Timestamp: {data['timestamp']}")

                    # Print feature info if available
                    if "feature_info" in data:
                        feature_info = data["feature_info"]
                        print(f"  - Total features: {feature_info['total_features']}")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return all_data

    def create_visualization_frame(self, frame_data, filename=""):
        """
        Create a visualization frame with landmarks and connections
        """
        # Create a black background
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        # Add filename and frame info as text
        cv2.putText(
            frame,
            f"Sequence: {filename}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Frame: {self.current_frame_idx + 1}/{self.total_frames}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        # Draw pose landmarks and connections
        if frame_data.get("pose") and frame_data["pose"].get("landmarks"):
            pose_landmarks = frame_data["pose"]["landmarks"]

            # Convert landmarks to pixel coordinates
            pose_points = []
            for landmark in pose_landmarks:
                x = int(landmark[0] * self.frame_width)
                y = int(landmark[1] * self.frame_height)
                pose_points.append((x, y))

                # Draw landmark point
                cv2.circle(frame, (x, y), 3, self.pose_landmark_color, -1)

            # Draw pose connections using stored connection data
            for connection in self.pose_connections:
                start_idx, end_idx = connection
                if start_idx < len(pose_points) and end_idx < len(pose_points):
                    cv2.line(
                        frame,
                        pose_points[start_idx],
                        pose_points[end_idx],
                        self.pose_connection_color,
                        2,
                    )

            # Display pose connection features if available
            if "connection_features" in frame_data["pose"]:
                connection_features = frame_data["pose"]["connection_features"]
                cv2.putText(
                    frame,
                    f"Pose connections: {len(connection_features)}",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

        # Draw hand landmarks and connections
        if frame_data.get("hands"):
            hand_count = 0
            for hand_data in frame_data["hands"]:
                hand_landmarks = hand_data["landmarks"]
                handedness = hand_data.get("handedness", "Unknown")

                # Convert landmarks to pixel coordinates
                hand_points = []
                for landmark in hand_landmarks:
                    x = int(landmark[0] * self.frame_width)
                    y = int(landmark[1] * self.frame_height)
                    hand_points.append((x, y))

                # Draw hand connections first (so they appear under landmarks)
                for connection in self.hand_connections:
                    start_idx, end_idx = connection
                    if start_idx < len(hand_points) and end_idx < len(hand_points):
                        cv2.line(
                            frame,
                            hand_points[start_idx],
                            hand_points[end_idx],
                            self.hand_connection_color,
                            2,
                        )

                # Draw hand landmarks
                for i, (x, y) in enumerate(hand_points):
                    cv2.circle(frame, (x, y), 4, self.hand_landmark_color, -1)
                    # Optionally draw landmark numbers for key points
                    if i in [0, 4, 8, 12, 16, 20]:  # Wrist and fingertips
                        cv2.putText(
                            frame,
                            str(i),
                            (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (255, 255, 255),
                            1,
                        )

                # Add hand label
                if hand_points:
                    label_pos = (hand_points[0][0], hand_points[0][1] - 20)
                    cv2.putText(
                        frame,
                        f"{handedness} Hand",
                        label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.hand_landmark_color,
                        2,
                    )

                # Display hand connection features if available
                if "connection_features" in hand_data:
                    connection_features = hand_data["connection_features"]
                    cv2.putText(
                        frame,
                        f"{handedness} hand connections: {len(connection_features)}",
                        (10, 100 + hand_count * 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    hand_count += 1

        # Display total connection features for the frame
        if "connection_features" in frame_data:
            total_hand_connections = len(
                frame_data["connection_features"].get("hands", [])
            )
            pose_connections_count = len(
                frame_data["connection_features"].get("pose", [])
            )

            cv2.putText(
                frame,
                f"Total connections - Hands: {total_hand_connections}, Pose: {pose_connections_count}",
                (10, self.frame_height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
            )

        return frame

    def show_controls(self):
        """
        Display control instructions
        """
        print("\n" + "=" * 60)
        print("LANDMARK VISUALIZER CONTROLS:")
        print("=" * 60)
        print("FRAME NAVIGATION:")
        print("  n/N          : Next frame")
        print("  p/P          : Previous frame")
        print("  r/R          : Reset to first frame of current file")
        print("  SPACE        : Toggle play/pause (auto-advance frames)")
        print("")
        print("FILE NAVIGATION:")
        print("  f/F          : Next landmark sequence file")
        print("  b/B          : Previous landmark sequence file")
        print("  HOME         : Go to first file")
        print("  END          : Go to last file")
        print("")
        print("OTHER CONTROLS:")
        print("  s/S          : Save current frame as image")
        print("  i/I          : Show file info")
        print("  c/C          : Show connection info")
        print("  h/H/?        : Show this help")
        print("  q/Q/ESC      : Quit")
        print("=" * 60)

    def show_file_info(self, all_data, current_file_idx):
        """
        Display information about loaded files
        """
        print("\n" + "=" * 60)
        print("LOADED LANDMARK SEQUENCE FILES:")
        print("=" * 60)
        for i, data in enumerate(all_data):
            marker = " >>> " if i == current_file_idx else "     "
            frames_count = len(data["data"]["frames"])
            timestamp = data["data"].get("timestamp", "Unknown")
            print(f"{marker}[{i+1:2d}] {data['filename']}")
            print(f"         Frames: {frames_count}, Timestamp: {timestamp}")

            # Show feature info if available
            if "feature_info" in data["data"]:
                feature_info = data["data"]["feature_info"]
                print(f"         Features: {feature_info['total_features']}")
        print("=" * 60)

    def show_connection_info(self):
        """
        Display information about connections
        """
        print("\n" + "=" * 60)
        print("CONNECTION INFORMATION:")
        print("=" * 60)
        print(f"Hand connections loaded: {len(self.hand_connections)}")
        print(f"Pose connections loaded: {len(self.pose_connections)}")

        if self.landmarks_data and "feature_info" in self.landmarks_data:
            feature_info = self.landmarks_data["feature_info"]
            print("\nFeature Dimensions:")
            print(f"  Hand landmarks: {feature_info['hand_landmarks']}")
            print(f"  Hand connections: {feature_info['hand_connections']}")
            print(f"  Pose landmarks: {feature_info['pose_landmarks']}")
            print(f"  Pose connections: {feature_info['pose_connections']}")
            print(f"  Total features: {feature_info['total_features']}")
        print("=" * 60)

    def run(self):
        """
        Main visualization loop
        """
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

        # Show controls and file info
        self.show_controls()
        print(f"\nLoaded {len(all_data)} landmark sequence files")
        self.show_file_info(all_data, 0)

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
            self.current_frame_idx = max(
                0, min(self.current_frame_idx, self.total_frames - 1)
            )

            # Get current frame data
            if self.total_frames > 0:
                frame_data = self.landmarks_data["frames"][self.current_frame_idx]

                # Create visualization
                vis_frame = self.create_visualization_frame(frame_data, filename)

                # Add file navigation info
                sequence_info = f"Sequence: {current_file_idx + 1}/{len(all_data)}"
                cv2.putText(
                    vis_frame,
                    sequence_info,
                    (10, self.frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

                # Add playback status if playing
                if playing:
                    cv2.putText(
                        vis_frame,
                        "PLAYING",
                        (self.frame_width - 100, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                # Display frame
                cv2.imshow(self.window_name, vis_frame)

            # Handle keyboard input
            key = cv2.waitKey(30 if playing else 0) & 0xFF

            if key in [ord("q"), ord("Q"), 27]:  # q, Q, or ESC
                break
            elif key in [ord("n"), ord("N")]:  # Next frame
                self.current_frame_idx = min(
                    self.current_frame_idx + 1, self.total_frames - 1
                )
            elif key in [ord("p"), ord("P")]:  # Previous frame
                self.current_frame_idx = max(self.current_frame_idx - 1, 0)
            elif key in [ord("f"), ord("F")]:  # Next landmark sequence file
                if current_file_idx < len(all_data) - 1:
                    current_file_idx += 1
                    self.current_frame_idx = 0
                    print(
                        f"Switched to sequence: {all_data[current_file_idx]['filename']}"
                    )
                else:
                    print("Already at last sequence file")
            elif key in [ord("b"), ord("B")]:  # Previous landmark sequence file
                if current_file_idx > 0:
                    current_file_idx -= 1
                    self.current_frame_idx = 0
                    print(
                        f"Switched to sequence: {all_data[current_file_idx]['filename']}"
                    )
                else:
                    print("Already at first sequence file")
            elif key == 0:  # HOME key
                if current_file_idx != 0:
                    current_file_idx = 0
                    self.current_frame_idx = 0
                    print(
                        f"Switched to first sequence: {all_data[current_file_idx]['filename']}"
                    )
            elif key == 1:  # END key
                last_idx = len(all_data) - 1
                if current_file_idx != last_idx:
                    current_file_idx = last_idx
                    self.current_frame_idx = 0
                    print(
                        f"Switched to last sequence: {all_data[current_file_idx]['filename']}"
                    )
            elif key in [ord("r"), ord("R")]:  # Reset to first frame of current file
                self.current_frame_idx = 0
                print(f"Reset to first frame of current sequence")
            elif key == ord(" "):  # Toggle play/pause
                playing = not playing
                status = "Playing" if playing else "Paused"
                print(
                    f"{status} - Current sequence: {all_data[current_file_idx]['filename']}"
                )
            elif key in [ord("i"), ord("I")]:  # Show file info
                self.show_file_info(all_data, current_file_idx)
            elif key in [ord("c"), ord("C")]:  # Show connection info
                self.show_connection_info()
            elif key in [ord("h"), ord("H"), ord("?")]:  # Show help
                self.show_controls()
            elif key in [ord("s"), ord("S")]:  # Save frame
                if self.total_frames > 0:
                    timestamp = self.landmarks_data.get("timestamp", "unknown")
                    save_filename = f"sequence_{current_file_idx}_frame_{self.current_frame_idx}_{timestamp}.png"
                    cv2.imwrite(save_filename, vis_frame)
                    print(f"Saved frame as {save_filename}")

            # Auto-advance frame if playing
            if playing and self.total_frames > 0:
                self.current_frame_idx += 1
                if self.current_frame_idx >= self.total_frames:
                    # Move to next sequence file or stop playing
                    if current_file_idx < len(all_data) - 1:
                        current_file_idx += 1
                        self.current_frame_idx = 0
                        print(
                            f"Auto-switched to next sequence: {all_data[current_file_idx]['filename']}"
                        )
                    else:
                        playing = False
                        print("Reached end of all landmark sequences. Stopped playing.")

        cv2.destroyAllWindows()


def main():
    print("Starting Interactive Landmark Visualizer...")
    visualizer = InteractiveLandmarkVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()
