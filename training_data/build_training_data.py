import re


changelog_file = "./unprocessed_text/text_processing_changelog.txt"
labeled_data_file = "./training_text/labeled_training_data.txt"
dialogue_file = "./training_text/darcy_dialogue_only.txt"
actions_file = "./training_text/darcy_actions_only.txt"
impressions_file = "./training_text/others_impressions_only.txt"

dialogue_and_actions = []
others_impressions = []

# extract from changelog
with open(changelog_file, "r", encoding="utf-8") as infile:
    for line in infile:
        line = line.strip()
        if line.startswith("<darcy-dialogue>") or line.startswith("<darcy-actions>"):
            dialogue_and_actions.append(line)
        elif line.startswith("<others-impressions-of-darcy>"):
            others_impressions.append(line)

# save into labeled_training_data.txt
with open(labeled_data_file, "w", encoding="utf-8") as outfile:
    for line in dialogue_and_actions + others_impressions:
        outfile.write(line + "\n")

# isolate categories into separate files
with open(labeled_data_file, "r", encoding="utf-8") as infile, \
     open(dialogue_file, "w", encoding="utf-8") as d_out, \
     open(actions_file, "w", encoding="utf-8") as a_out, \
     open(impressions_file, "w", encoding="utf-8") as i_out:
    
    for line in infile:
        line = line.strip()
        if line.startswith("<darcy-dialogue>"):
            d_out.write(line + "\n")
        elif line.startswith("<darcy-actions>"):
            a_out.write(line + "\n")
        elif line.startswith("<others-impressions-of-darcy>"):
            i_out.write(line + "\n")

print("\nExtraction and categorization complete.\n")
