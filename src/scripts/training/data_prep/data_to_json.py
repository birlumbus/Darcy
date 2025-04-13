import json
import re


'''
Opens txt datasets built by build_dataset scripts
Writes their contents into json files
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
    in_1  = "./processed_data/datasets_txt/dataset_1.txt"
    in_2  = "./processed_data/datasets_txt/dataset_2.txt"
    out_1 = "./processed_data/datasets_json/dataset_1.json"
    out_2 = "./processed_data/datasets_json/dataset_2.json"

    data_to_json(in_1, out_1)
    data_to_json(in_2, out_2)


if __name__ == "__main__":
    main()
