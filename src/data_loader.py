import os
import json
import pandas as pd
from tqdm import tqdm

def load_dataset(base_path: str):

    data = []

    for subtopic in os.listdir(base_path):
        subfolder = os.path.join(base_path, subtopic)
        if not os.path.isdir(subfolder):
            continue

        for file in tqdm(os.listdir(subfolder), desc=f"Loading {subtopic}"):
            if not file.endswith(".json"):
                continue
            filepath = os.path.join(subfolder, file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    item = json.load(f)
                    # The MATH dataset uses 'problem' for question text
                    text = item.get("problem", "")
                    label = item.get("type", subtopic)
                    if text.strip():
                        data.append({"question_text": text, "label": label})
            except Exception as e:
                print(f" Skipping file {file}: {e}")
                continue

    return pd.DataFrame(data)

