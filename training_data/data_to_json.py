import json
import re


input_file = "./training_text/labeled_training_data.txt"
output_file = "./training_text/labeled_training_data.json"

structured_data = []

pattern = re.compile(r"<(.*?)>\s*(.*)")

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.match(line.strip())
        if match:
            category, content = match.groups()
            structured_data.append({"category": category, "content": content})

# Save the structured data as JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(structured_data, f, ensure_ascii=False, indent=4)

print(f"Structured JSON data saved to {output_file}")
