import re
from collections import defaultdict


def count_labels(input_file, output_file):
    label_counts = defaultdict(int)
    
    # process
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"^<([^>]+)>", line)
            if match:
                label = match.group(1)
                label_counts[label] += 1

    # write to output
    with open(output_file, "w", encoding="utf-8") as f:
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{label}: {count}\n")
