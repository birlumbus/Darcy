import json
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import string
# if running standalone, use "import ref_dialogue_capture"
from . import ref_dialogue_capture
# import ref_dialogue_capture


# nltk.download('punkt')
# nltk.download('wordnet')


def preprocess(text):
    """
    Lowercase the text, tokenize it, and remove punctuation.
    """
    tokens = nltk.word_tokenize(text.lower())
    return [t for t in tokens if t not in string.punctuation]


def tokenize_if_needed(text):
    """
    If the input is already tokenized (a list), return it as is.
    If it's a string, tokenize it using preprocess.
    """
    if isinstance(text, list):
        return text
    elif isinstance(text, str):
        return preprocess(text)
    else:
        raise ValueError("Input text must be either a string or a list of tokens.")


def lcs_length(x, y):
    """
    Compute the length of the longest common subsequence between two token lists.
    """
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    return dp[m][n]


def rouge_l(candidate_tokens, reference_tokens):
    """
    Compute ROUGE-L F1 score between candidate and reference tokens.
    """
    lcs = lcs_length(candidate_tokens, reference_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def evaluate_texts(ref_texts, candidate_text):
    """
    Evaluate a candidate text against a set of reference texts using BLEU, ROUGE-L, and METEOR.
    Supports input that is either already tokenized (list of tokens) or a string.
    """
    smoothie = SmoothingFunction().method1

    # tokenize references and candidate text if needed
    tokenized_refs = [tokenize_if_needed(ref) for ref in ref_texts]
    candidate_tokens = tokenize_if_needed(candidate_text)
    
    # compute BLEU scores
    bleu1 = corpus_bleu([tokenized_refs], [candidate_tokens], weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = corpus_bleu([tokenized_refs], [candidate_tokens], weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu4 = corpus_bleu([tokenized_refs], [candidate_tokens], smoothing_function=smoothie)
    
    # compute ROUGE-L (average maximum score among references)
    rouge_scores = [rouge_l(candidate_tokens, tokenize_if_needed(ref)) for ref in ref_texts]
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    
    # compute METEOR (compares meaning more than style)
    meteor = meteor_score(tokenized_refs, candidate_tokens)
    
    return {
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu4": bleu4,
        "rouge_l": avg_rouge,
        "meteor": meteor
    }


def process_results(json_file, ref_file_path):
    """
    Process the JSON file with prompt outputs by:
    - Evaluating each candidate against a set of reference texts.
    - For non-base outputs (version != "0"), also evaluating against the base output.
    - Adding the evaluation results back into the JSON structure.
    """
    # capture references
    references = ref_dialogue_capture.capture_references(ref_file_path, min_words=12, max_words=80, sample_size=100)
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for prompt in data:
        # group outputs by model to easily locate base output
        outputs_by_model = {}
        for output in prompt["outputs"]:
            if output.get('output') != '[[no output]]':
                model = output["model"]
                outputs_by_model.setdefault(model, []).append(output)
        
        for model, outputs in outputs_by_model.items():
            # identify baseline output for this model
            baseline = next((o for o in outputs if str(o.get("version")) == "0"), None)
            
            for output in outputs:
                candidate_text = output["output"]
                # always evaluate against external references
                ref_scores = evaluate_texts(references, candidate_text)
                output["evaluation_vs_references"] = ref_scores
                
                # for non-base outputs, evaluate against baseline (if available)
                if str(output.get("version")) != "0" and baseline is not None:
                    baseline_text = baseline["output"]
                    baseline_scores = evaluate_texts([baseline_text], candidate_text)
                    output["evaluation_vs_baseline"] = baseline_scores
    
    # write updated results back to JSON file
    print("Updating JSON file with evaluation metrics...")
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    json_file = './prompt_results/json/prompt_results_6.json'
    ref_file_path = '../training_data/training_text/isolated_categories/darcy_dialogue_only.txt'
    print("\nEvaluating...")
    process_results(json_file, ref_file_path)
    print("Done.\n")
