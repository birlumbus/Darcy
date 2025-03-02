import re


def extract_dialogue(input_file, output_file):
    dialogue_pattern = re.compile(r'[“](.*?)[”]', re.DOTALL)
    
    # open
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()  # read file as single string
    
    # replace paragraph breaks within quotes with a space
    text = re.sub(r'[“](.*?)[”]', lambda m: m.group(0).replace('\n', ' '), text, flags=re.DOTALL)
    
    # find
    dialogues = dialogue_pattern.findall(text)
    
    # save
    with open(output_file, 'w', encoding='utf-8') as f:
        for quote in dialogues:
            f.write(quote.strip() + "\n")


input_file = "./unprocessed_text/pride_and_prejudice.txt"
output_file = "./unprocessed_text/all_dialogue.txt"
extract_dialogue(input_file, output_file)
