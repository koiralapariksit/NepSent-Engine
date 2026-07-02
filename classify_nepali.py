from pathlib import Path
from phase3_muril_finetuning import MuRILModel

# Load the fine-tuned MuRIL model using the project dataset path
model = MuRILModel(Path("d:/ds mdoel/data/04_annotated_dataset.jsonl"))

sentence = "म धेरै खुसी छु।"
label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}

prediction = model.predict([sentence])[0]
print(f"Input: {sentence}")
print(f"Predicted sentiment: {label_map[prediction]}")
