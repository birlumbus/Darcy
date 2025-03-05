import json
import re


'''
Opens text files built by build_training_data scripts
Writes their contents into json files in the same location
'''


def data_to_json(in_file, out_file):
    structured_data = []
    pattern = re.compile(r"<(.*?)>\s*(.*)")
    
    # input
    with open(in_file, "r", encoding="utf-8") as f:
        # process
        for line in f:
            match = pattern.match(line.strip())
            if match:
                category, content = match.groups()
                structured_data.append({"category": category, "content": content})
    
    #output
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=4)
        

def main():
    in_1  = "./training_text/final_txt/labeled_training_data_1.txt"
    out_1 = "./training_text/final_json/labeled_training_data_1.json"
    in_2  = "./training_text/final_txt/labeled_training_data_2.txt"
    out_2 = "./training_text/final_json/labeled_training_data_2.json"

    data_to_json(in_1, out_1)
    data_to_json(in_2, out_2)


if __name__ == "__main__":
    main()
