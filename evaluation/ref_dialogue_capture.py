import random


# load dialogues from your file
def load_dialogues(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # strip tags and whitespace
    dialogues = [line.replace('<darcy-dialogue>', '').strip() for line in lines]
    return dialogues


# select dialogue of specified length
def capture_dialogues(dialogues, min_words=8, max_words=20):
    suitable_references = [d for d in dialogues if min_words <= len(d.split()) <= max_words]
    return random.sample(suitable_references)


# save dialogue to new file
def save_references


if __name__ == '__main__':
    all_dialogue_txt = '../training_data/training_text/isolated_categories/darcy_dialogue_only.txt'
    references_txt = '../training_data/training_text/isolated_categories/references_for_analysis.txt'
    dialogues = load_dialogues(filepath)
    
    selected_samples = capture_dialogues(dialogues, min_words=8, max_words=20)

    print('\nSaving references')
    save_references(references_txt)
    print('...done.\n')
