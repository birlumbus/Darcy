import re
from collections import defaultdict

input_file = "./training_text/labeled_training_data.txt"  # Replace with the actual file name
output_file = "./training_text/tag_counts.txt"

tag_counts = defaultdict(int)

# process
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        match = re.match(r"^<([^>]+)>", line)
        if match:
            # store
            tag = match.group(1)
            tag_counts[tag] += 1

# write to output
with open(output_file, "w", encoding="utf-8") as f:
    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{tag}: {count}\n")

print(f"Tag counts saved to {output_file}")
