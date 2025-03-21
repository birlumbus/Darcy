import re
import json
import argparse


def read_file(filepath):
    """Read the entire contents of the file."""
    with open(filepath, "r", encoding="utf8") as f:
        return f.read()


def split_sections(text):
    """Split the file into sections based on 'Prompt text:' markers."""
    return [sec for sec in re.split(r"Prompt text:\s*", text) if sec.strip()]


def extract_prompt(section):
    """Extract the prompt text from a section (first block of non-empty lines)."""
    lines = section.splitlines()
    prompt_lines = []
    for line in lines:
        if not line.strip():
            break
        prompt_lines.append(line.strip())
    return " ".join(prompt_lines)


def extract_perplexity(sub_text):
    """Extract the perplexity value from a block of text."""
    match = re.search(r"Perplexity:\s*([0-9\.eE+-]+)", sub_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def extract_outputs(section):
    """
    Find all outputs in a section. Each output is identified by lines like
    'From medium-0:' and its corresponding block.
    """
    output_pattern = re.compile(r"From\s+(\w+)-([\w\.]+):")
    outputs = []
    matches = list(output_pattern.finditer(section))
    
    for i, match in enumerate(matches):
        model = match.group(1)
        version = match.group(2)
        start = match.end()
        # determine end position by next match or end of section
        end = matches[i+1].start() if i+1 < len(matches) else len(section)
        sub_text = section[start:end]
        perplexity = extract_perplexity(sub_text)
        # extract the model's output text by taking text before "Perplexity:" marker.
        if "Perplexity:" in sub_text:
            output_text = sub_text.split("Perplexity:")[0].strip()
        else:
            output_text = sub_text.strip()
        
        outputs.append({
            "model": model,
            "version": version,
            "output": output_text,
            "perplexity": perplexity
        })
    return outputs


def sort_outputs(outputs):
    """Sort the outputs by model title and then by version."""
    return sorted(outputs, key=lambda x: (x["model"], x["version"]))


def parse_sections(sections):
    """Parse each section to extract the prompt and its outputs."""
    results = []
    for sec in sections:
        prompt = extract_prompt(sec)
        outputs = sort_outputs(extract_outputs(sec))
        results.append({
            "prompt": prompt,
            "outputs": outputs
        })
    return results


def parse_file(filepath):
    """Parse the file into structured JSON objects."""
    text = read_file(filepath)
    sections = split_sections(text)
    return parse_sections(sections)


def main():
    parser = argparse.ArgumentParser(
        description="Parse GPT model test results into JSON using a prompt_results integer identifier."
    )
    parser.add_argument("input_integer", type=int,
                        help="An integer to identify the prompt results file. The input file is constructed as './prompt_results/txt/prompt_results_<input_integer>.txt'")
    args = parser.parse_args()

    input_filepath = f"./prompt_results/txt/prompt_results_{args.input_integer}.txt"
    output_filepath = f"./prompt_results/json/prompt_results_{args.input_integer}.json"
    
    print('\nParsing data...')
    parsed_data = parse_file(input_filepath)
    
    print('Writing to JSON file...')
    with open(output_filepath, "w", encoding="utf8") as out_f:
        json.dump(parsed_data, out_f, indent=2)
    print('...Done\n')


if __name__ == "__main__":
    main()
