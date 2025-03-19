# load dialogues from file
def load_dialogue(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # strip tags and whitespace
    dialogue = [line.replace('<darcy-dialogue>', '').strip() for line in lines]
    return dialogue


# select dialogue of specified length
def capture_references(dialogues, min_words=15, max_words=80):
    suitable_references = [d for d in dialogues if min_words <= len(d.split()) <= max_words]
    return suitable_references


# save dialogue to new file
def save_references(filepath, selected_samples):
    content = "\n".join(selected_samples)
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content)


if __name__ == '__main__':
    all_dialogue = '../training_data/training_text/isolated_categories/darcy_dialogue_only.txt'
    references = './training_data/training_text/isolated_categories/references_for_analysis.txt'

    dialogue = load_dialogue(all_dialogue)
    selected_samples = capture_references(dialogue, min_words=8, max_words=20)

    print('\nSaving references')
    save_references(references, selected_samples)
    print('...done.\n')
