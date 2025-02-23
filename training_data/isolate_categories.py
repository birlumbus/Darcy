import re


input_file = "./training_text/labeled_training_data.txt"
dialogue_file = "./training_text/darcy_dialogue_only.txt"
actions_file = "./training_text/darcy_actions.txt"
impressions_file = "./training_text/others_impressions_only.txt"


with open(input_file, "r", encoding="utf-8") as infile, \
     open(dialogue_file, "w", encoding="utf-8") as d_out, \
     open(actions_file, "w", encoding="utf-8") as a_out, \
     open(impressions_file, "w", encoding="utf-8") as i_out:
    
    for line in infile:
        if line.startswith("<darcy-dialogue>"):
            d_out.write(line[len("<darcy-dialogue>"):].strip() + "\n")
        elif line.startswith("<darcy-actions>"):
            a_out.write(line[len("<darcy-actions>"):].strip() + "\n")
        elif line.startswith("<others-impressions-of-darcy>"):
            i_out.write(line[len("<others-impressions-of-darcy>"):].strip() + "\n")

print(f"Extracted dialogues have been saved to {dialogue_file}, {actions_file}, and {impressions_file}")
