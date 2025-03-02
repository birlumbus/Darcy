import re


# input files
changelog_file = "./unprocessed_text/text_processing_changelog.txt"
questions_file = "./training_text/questions.txt"

# output files
labeled_data_file = "./training_text/labeled_training_data.txt"
dialogue_file = "./training_text/darcy_dialogue_only.txt"
actions_file = "./training_text/darcy_actions_only.txt"
impressions_file = "./training_text/others_impressions_only.txt"


dialogue_and_actions = []
others_impressions = []
questions_and_answers = []

# extract from changelog
with open(changelog_file, "r", encoding="utf-8") as changelog:
    for line in changelog:
        line = line.strip()
        if line.startswith("<darcy-dialogue>") or line.startswith("<darcy-actions>"):
            dialogue_and_actions.append(line)
        elif line.startswith("<others-impressions-of-darcy>"):
            others_impressions.append(line)

# extract questions
with open(questions_file, "r", encoding="utf-8") as q_file:
    qa_lines = [line.strip() for line in q_file if line.strip()]
    
    for i in range(0, len(qa_lines) - 1, 2):
        question = f"<question> {qa_lines[i]}"
        answer = f"<answer> {qa_lines[i + 1]}"
        questions_and_answers.append(f"{question}\n{answer}")

# add others_impressions after halfway point
halfway_point = len(dialogue_and_actions) // 2
interval_oi = (len(dialogue_and_actions) - halfway_point) // max(1, len(others_impressions))
interleaved_data = dialogue_and_actions[:halfway_point]

for i, line in enumerate(dialogue_and_actions[halfway_point:], start=1):
    interleaved_data.append(line)
    if others_impressions and i % interval_oi == 0:
        interleaved_data.append(others_impressions.pop(0))

# add questions after halfway point
interval_qa = len(interleaved_data) // max(1, len(questions_and_answers))
final_data = []

for i, line in enumerate(interleaved_data):
    final_data.append(line)
    if questions_and_answers and i % interval_qa == 0:
        final_data.append(questions_and_answers.pop(0))

# save into labeled_training_data.txt
with open(labeled_data_file, "w", encoding="utf-8") as outfile:
    for line in final_data:
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
