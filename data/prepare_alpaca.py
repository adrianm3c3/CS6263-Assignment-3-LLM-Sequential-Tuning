from datasets import load_dataset
import json

dataset = load_dataset("yahma/alpaca-cleaned")

data = dataset["train"]

formatted = []

for example in data:
    formatted.append({
        "instruction": example["instruction"],
        "input": example["input"],
        "output": example["output"]
    })

with open("data/alpaca_dataset.json", "w") as f:
    json.dump(formatted, f, indent=2)

print("Saved Alpaca dataset")