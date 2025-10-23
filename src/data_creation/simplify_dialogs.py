import csv
import json
from ..audio2gloss import AudioToGlossConverter


input_path = "data/dataset/MediTOD/dialogs.json"
output_path_json = "data/dataset/MediTOD/dialog_simplified.json"
output_path_csv = "data/dataset/MediTOD/utterances.csv"

converter = AudioToGlossConverter()
converter.load_model()


def simplify_dialog():
    # Read the original dialog.json file
    with open(input_path, "r") as f:
        data = json.load(f)

    # Extract all utterances with just the text field
    with open(output_path_csv, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["gloss", "text"])

        utterances = []
        for _, dialog_data in data.items():
            if "utterances" in dialog_data:
                for utterance in dialog_data["utterances"]:
                    text = utterance["text"]
                    utterances.append({"text": text})
                    clause_glosses_list = converter.text_to_glosses(text)
                    if clause_glosses_list and any(len(clause) > 0 for clause in clause_glosses_list):
                        flattened_gloss_list = [
                            gloss for clause in clause_glosses_list for gloss in clause
                        ]
                        gloss_sequence = " ".join(flattened_gloss_list)
                        csv_writer.writerow([gloss_sequence, text])
                    else:
                        continue

    # Create simplified structure
    simplified = {"utterances": utterances}

    # Write to new file
    with open(output_path_json, "w") as f:
        json.dump(simplified, f, indent=2)

    print(f"Simplified {len(utterances)} utterances")
    print("Output saved to dialog_simplified.json")


if __name__ == "__main__":
    simplify_dialog()
