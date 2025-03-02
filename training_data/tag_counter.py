import re
from collections import defaultdict


def count_tags(input_file, output_file):
    tag_counts = defaultdict(int)
    
    # process
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"^<([^>]+)>", line)
            if match:
                tag = match.group(1)
                tag_counts[tag] += 1

    # write to output
    with open(output_file, "w", encoding="utf-8") as f:
        for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{tag}: {count}\n")
