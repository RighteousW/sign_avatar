import json
import os
import re

# Load representatives file
with open(
    "data/output/gesture_metadata/gesture_representatives_metadata.json", "r"
) as f:
    data = json.load(f)

representatives = data["representatives"]

# Scan gloss_videos folders
for folder_name in os.listdir("data/gloss_videos"):
    folder_path = os.path.join("data/gloss_videos", folder_name)
    if not os.path.isdir(folder_path):
        continue

    # Get one file from folder
    files = [f for f in os.listdir(folder_path) if f.endswith(".avi")]
    if not files:
        continue

    # Extract base gloss from filename
    filename = files[0]
    base = re.sub(r"_\d{8}_\d{6}(_flipped)?\.avi$", "", filename)

    # Find source entry in representatives
    if base not in representatives:
        continue

    source = representatives[base]

    # Add folder name entry
    if folder_name not in representatives:
        representatives[folder_name] = source.copy()

    # If base contains _or, add split entries
    if "_or" in base:
        parts = base.split("_or_")
        for part in parts:
            if part and part not in representatives:
                representatives[part] = source.copy()

# Save updated file
with open(
    "data/output/gesture_metadata/gesture_representatives_metadata.json", "w"
) as f:
    json.dump(data, f, indent=2)

print("Done!")
