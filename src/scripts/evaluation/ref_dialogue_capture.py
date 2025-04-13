import random


'''
Pulls sample of reference dialogues for testing outputs against
Default sample size is 100
They're saved to ref_dialogue folder
'''


# load dialogues from file
def load_dialogue(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # strip tags and whitespace
    dialogue = [line.replace('<darcy-dialogue>', '').strip() for line in lines]
    return dialogue


# select dialogue of specified length
def capture_references(filepath, min_words, max_words, sample_size):
    dialogue = load_dialogue(filepath)
    candidate_references = [d for d in dialogue if min_words <= len(d.split()) <= max_words]
    sampled_references = random.sample(candidate_references, sample_size)
    return sampled_references


# save dialogue to new file (TESTING ONLY)
def save_references(filepath, selected_samples):
    content = "\n".join(selected_samples)
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content)


if __name__ == '__main__':
    all_dialogue = '../../training_data/processed_data/data_groups/darcy_dialogue.txt'
    # TESTING ONLY (add filepath of choice)
    # out_file = ''

    selected_samples = capture_references(all_dialogue, min_words=12, max_words=28, sample_size=100)

    # print('\nSaving references')
    # save_references(out_file, selected_samples)
    print('...done.\n')