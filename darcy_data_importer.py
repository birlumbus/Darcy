import re

def extract_dialogue(input_file, output_file):
    dialogue_pattern = re.compile(r'[“](.*?)[”]', re.DOTALL)
    
    # Open
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()  # Read the entire file as a single string
    
    # Replace paragraph breaks (double newlines) within quotes with a space
    text = re.sub(r'[“](.*?)[”]', lambda m: m.group(0).replace('\n', ' '), text, flags=re.DOTALL)
    
    # Find
    dialogues = dialogue_pattern.findall(text)
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        for quote in dialogues:
            f.write(quote.strip() + "\n")


input_file = "/Users/rhodri/Projects/Darcy/pride_and_prejudice.txt"
output_file = "/Users/rhodri/Projects/Darcy/all_dialogue.txt"
extract_dialogue(input_file, output_file)
print(f"Extracted dialogue saved to {output_file}")
