import re


def separate_sentences(input, output):
    with open(input, 'r') as file:
        text = file.read()
    
    # splits at periods, question marks, and exclamation marks followed by spaces
    sentences = re.split(r'(?<=\.|\?|!)\s+', text)
    
    # tag each line
    with open(output, 'w') as file:
        for sentence in sentences:
            file.write(f"<darcy-dialogue> {sentence.strip()}\n")


input = './unprocessed_text/darcy_letter.txt'
output = './unprocessed_text/darcy_letter_as_dialogue.txt'

separate_sentences(input, output)
