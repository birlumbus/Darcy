import random


input_file = './unprocessed_text/text_processing_changelog.txt'
output_file = './training_text/labeled_training_data.txt'
dialogue_and_actions = []
others_impressions = []


with open(input_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        line = line.strip()
        if line.startswith("<darcy-dialogue>") or line.startswith("<darcy-actions>"):
            dialogue_and_actions.append(line)
        elif line.startswith("<others-impressions-of-darcy>"):
            others_impressions.append(line)


random.shuffle(others_impressions)

# determine start point for insertions of others_impressions
halfway_point = len(dialogue_and_actions) // 2
result_lines = dialogue_and_actions[:halfway_point]

# determine interval by which others_impressions will be separated
interval = len(dialogue_and_actions[halfway_point:]) // len(others_impressions)
others_index = 0


for i, line in enumerate(dialogue_and_actions[halfway_point:]):
    if others_index < len(others_impressions) and i % interval == 0:
        result_lines.append(others_impressions[others_index])
        others_index += 1
    result_lines.append(line)

# add remaining others_impressions
while others_index < len(others_impressions):
    result_lines.append(others_impressions[others_index])
    others_index += 1


with open(output_file, 'w', encoding='utf-8') as outfile:
    outfile.write('\n'.join(result_lines) + '\n')

