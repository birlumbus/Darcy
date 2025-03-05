import json
import re


input_file = "./training_text/labeled_training_data_2.txt"
output_file = "./training_text/labeled_training_data_2.json"

structured_data = []
pattern = re.compile(r"<(.*?)>\s*(.*)")


with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.match(line.strip())
        if match:
            category, content = match.groups()
            structured_data.append({"category": category, "content": content})


with open(output_file, "w", encoding="utf-8") as f:
    json.dump(structured_data, f, ensure_ascii=False, indent=4)
