import label_counter


'''
Builds dataset_1.txt using dialogue, question/answer pairs, actions, & impressions
    1. Dialogue and actions are added to a list
    2. Question/answer pairs are interleaved equally throughout
    3. Impressions are shuffled and added at equal intervals after the halfway point.
'''


# input files
changelog_file = "./unprocessed_data/processing_changelogs/dataset_1_changelog.txt"
questions_file = "./processed_data/data_groups/supplemental_qa.txt"

# save locations for each data group
dialogue_file = "./processed_data/data_groups/darcy_dialogue.txt"
actions_file = "./processed_data/data_groups/darcy_actions.txt"
impressions_file = "./processed_data/data_groups/others_impressions.txt"

# output files
labeled_data_file = "./processed_data/datasets_txt/dataset_1.txt"
label_count_file = "./processed_data/label_totals/label_counts_1.txt"

dialogue_and_actions = []
others_impressions = []
questions_and_answers = []


def process_changelog():
    with open(changelog_file, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if line.startswith("<darcy-dialogue>") or line.startswith("<darcy-actions>"):
                dialogue_and_actions.append(line)
            elif line.startswith("<others-impressions-of-darcy>"):
                others_impressions.append(line)


def process_questions():
    with open(questions_file, "r", encoding="utf-8") as qfile:
        qa_lines = [line.strip() for line in qfile if line.strip()]
        
        for i in range(0, len(qa_lines) - 1, 2):
            question = f"<question> {qa_lines[i]}"
            answer = f"<answer> {qa_lines[i + 1]}"
            questions_and_answers.append(f"{question}\n{answer}")


def interleave_impressions():
    halfway_point = len(dialogue_and_actions) // 2
    interval_oi = (len(dialogue_and_actions) - halfway_point) // max(1, len(others_impressions))
    interleaved_data = dialogue_and_actions[:halfway_point]

    for i, line in enumerate(dialogue_and_actions[halfway_point:], start=1):
        interleaved_data.append(line)
        if others_impressions and i % interval_oi == 0:
            interleaved_data.append(others_impressions.pop(0))
    return interleaved_data


def interleave_questions(interleaved_data):
    interval_qa = len(interleaved_data) // max(1, len(questions_and_answers))
    final_data = []

    for i, line in enumerate(interleaved_data):
        final_data.append(line)
        if questions_and_answers and i % interval_qa == 0:
            final_data.append(questions_and_answers.pop(0))
    return final_data


def save_labeled_data(final_data):
    with open(labeled_data_file, "w", encoding="utf-8") as outfile:
        for line in final_data:
            outfile.write(line + "\n")


def isolate_categories():
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


def main():
    print("Processing changelog...")
    process_changelog()
    print("Processing questions...")
    process_questions()
    print("Interleaving impressions...")
    interleaved_data = interleave_impressions()
    print("Interleaving questions...")
    final_data = interleave_questions(interleaved_data)
    print("Saving labeled data...")
    save_labeled_data(final_data)
    print("Isolating categories...")
    isolate_categories()
    print("Counting labels...")
    label_counter.count_labels(labeled_data_file, label_count_file)
    print("\nExtraction and categorization complete.\n")


if __name__ == "__main__":
    main()

