import os
import pickle
from tqdm import tqdm

from ..constants import (
    LANDMARKS_DIR_HANDS_ONLY,
    LANDMARKS_DIR_HANDS_POSE,
)


def convert_landmark_file_to_hands_only(input_path, output_path):
    """
    Convert a hand+pose landmark file to hands-only format
    """
    with open(input_path, "rb") as f:
        landmarks_data = pickle.load(f)

    # Update landmark types to only include hands
    landmarks_data["landmark_types"] = ["hand_landmarks"]

    # Update feature info
    landmarks_data["feature_info"] = {
        "hand_landmarks": 126,  # 21 * 3 * 2
        "pose_landmarks": 0,
        "total_features": 126,
        "max_hands": 2,
        "hand_landmarks_per_hand": 63,  # 21 * 3
        "pose_total_landmarks": 0,
        "pose_coords_per_landmark": 0,
    }
    landmarks_data["max_feature_vector_size"] = 126

    # Remove pose data from all frames
    for frame_data in landmarks_data["frames"]:
        if "pose" in frame_data:
            del frame_data["pose"]

    # Save to output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(landmarks_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def convert_all_landmarks():
    """
    Convert all hand+pose landmark files to hands-only format
    """
    if not os.path.exists(LANDMARKS_DIR_HANDS_POSE):
        print(f"Error: Source directory not found at {LANDMARKS_DIR_HANDS_POSE}")
        return

    os.makedirs(LANDMARKS_DIR_HANDS_ONLY, exist_ok=True)

    # Count total files
    total_files = 0
    files_by_folder = {}

    for folder_num in sorted(os.listdir(LANDMARKS_DIR_HANDS_POSE)):
        folder_path = os.path.join(LANDMARKS_DIR_HANDS_POSE, folder_num)
        if os.path.isdir(folder_path):
            pkl_files = sorted(
                [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
            )
            files_by_folder[folder_num] = pkl_files
            total_files += len(pkl_files)

    print(f"Converting {total_files} landmark files to hands-only format...")
    print(f"Source: {LANDMARKS_DIR_HANDS_POSE}")
    print(f"Destination: {LANDMARKS_DIR_HANDS_ONLY}\n")

    # Overall progress bar
    overall_pbar = tqdm(
        total=total_files, desc="Overall Progress", position=0, leave=True, unit="file"
    )

    success_count = 0
    failed_count = 0
    failed_files = []

    for folder_num in sorted(files_by_folder.keys()):
        input_folder_path = os.path.join(LANDMARKS_DIR_HANDS_POSE, folder_num)
        output_folder_path = os.path.join(LANDMARKS_DIR_HANDS_ONLY, folder_num)

        pkl_files = files_by_folder[folder_num]

        # Current folder progress bar
        folder_pbar = tqdm(
            pkl_files, desc=f"Folder {folder_num}", position=1, leave=False, unit="file"
        )

        for pkl_file in folder_pbar:
            input_file_path = os.path.join(input_folder_path, pkl_file)
            output_file_path = os.path.join(output_folder_path, pkl_file)

            folder_pbar.set_postfix_str(pkl_file)

            try:
                convert_landmark_file_to_hands_only(input_file_path, output_file_path)
                success_count += 1
            except Exception as e:
                failed_count += 1
                failed_files.append({"file": input_file_path, "error": str(e)})

            overall_pbar.update(1)

        folder_pbar.close()

    overall_pbar.close()

    # Print summary
    print(f"\n✓ Successfully converted {success_count} files")
    if failed_count > 0:
        print(f"✗ Failed to convert {failed_count} files")
        for failed in failed_files[:5]:  # Show first 5 failures
            print(f"  - {failed['file']}: {failed['error']}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")

    print(f"\nHands-only landmarks saved to: {LANDMARKS_DIR_HANDS_ONLY}")


def main():
    print("Starting conversion of hand+pose landmarks to hands-only format...")
    convert_all_landmarks()
    print("\nConversion complete!")


if __name__ == "__main__":
    main()
