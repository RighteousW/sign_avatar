import csv
import json

input_path = "data/dataset/MediTOD/dialogs.json"
output_path_json = "data/dataset/MediTOD/dialog_simplified.json"
output_path_csv = "data/dataset/MediTOD/utterances.csv"


def simplify_dialog():
    # Read the original dialog.json file
    with open(input_path, "r", encoding="utf-8") as f:  # Added encoding
        data = json.load(f)

    utterances = []  # Move outside the CSV context manager

    # Extract all utterances with just the text field
    with open(output_path_csv, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["text"])

        for dialog_id, dialog_data in data.items():  # Better variable name
            if "utterances" in dialog_data:
                for utterance in dialog_data["utterances"]:
                    if "text" in utterance:  # Safety check
                        text = utterance["text"]
                        utterances.append({"text": text})
                        csv_writer.writerow([text])

    # Create simplified structure
    simplified = {"utterances": utterances}

    # Write to new file
    with open(output_path_json, "w", encoding="utf-8") as f:  # Added encoding
        json.dump(simplified, f, indent=2, ensure_ascii=False)  # Added ensure_ascii

    print(f"Simplified {len(utterances)} utterances")
    print(f"Output saved to {output_path_json}")
    print(f"Utterances saved to {output_path_csv}")


if __name__ == "__main__":
    simplify_dialog()
