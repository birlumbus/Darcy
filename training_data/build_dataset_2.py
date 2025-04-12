import tag_counter


'''
Builds dataset_2.txt using prompt/dialogue & question/answer pairs
    1. Dialogue and actions are added to a list
    2. Question/answer pairs are interleaved equally throughout, ensuring they never interrupt a prompt/dialogue pair
'''


# input files
changelog_file = "./unprocessed_data/processing_changelogs/dataset_2_changelog.txt"
questions_file = "./processed_data/data_groups/supplemental_qa.txt"

# output files
labeled_data_file = "./processed_data/datasets_txt/dataset_2.txt"
tag_count_file = "./processed_data/label_totals/label_counts_2.txt"

prompts_and_dialogue = []
questions_and_answers = []


def process_changelog():
    with open(changelog_file, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if line.startswith("<user>") or line.startswith("<darcy-dialogue>"):
                line = line.replace("<darcy-dialogue>", "<darcy>")
                prompts_and_dialogue.append(line)


def process_questions():
    with open(questions_file, "r", encoding="utf-8") as qfile:
        qa_lines = [line.strip() for line in qfile if line.strip()]
        for i in range(0, len(qa_lines) - 1, 2):
            question = f"<question> {qa_lines[i]}"
            answer = f"<answer> {qa_lines[i + 1]}"
            questions_and_answers.append(f"{question}\n{answer}")


def interleave_questions():
    # group prompts_and_dialogue into pairs so that question/answer pairs are only inserted between complete groups.
    paired_prompts = []
    i = 0
    while i < len(prompts_and_dialogue):
        if (i + 1 < len(prompts_and_dialogue) and
            prompts_and_dialogue[i].startswith("<user>") and
            prompts_and_dialogue[i+1].startswith("<darcy>")):
            paired_prompts.append([prompts_and_dialogue[i], prompts_and_dialogue[i+1]])
            i += 2
        else:
            paired_prompts.append([prompts_and_dialogue[i]])
            i += 1

    num_pairs = len(paired_prompts)
    # Calculate an interval based on the number of pairs.
    interval_qa = num_pairs // max(1, len(questions_and_answers))

    final_data = []
    for idx, pair in enumerate(paired_prompts):
        # Append the entire prompt/dialogue pair (or single line if not a full pair)
        final_data.extend(pair)
        # Insert a question/answer pair after complete pairs at the calculated intervals
        if questions_and_answers and (idx + 1) % interval_qa == 0:
            final_data.append(questions_and_answers.pop(0))
    return final_data


def save_labeled_data(final_data):
    with open(labeled_data_file, "w", encoding="utf-8") as outfile:
        for line in final_data:
            outfile.write(line + "\n")


def main():
    print("Processing changelog...")
    process_changelog()
    print("Processing questions...")
    process_questions()
    print("Interleaving questions...")
    final_data = interleave_questions()
    print("Saving labeled data...")
    save_labeled_data(final_data)
    print("Counting tags...")
    tag_counter.count_tags(labeled_data_file, tag_count_file)
    print("\nExtraction and categorization complete.\n")


if __name__ == "__main__":
    main()
