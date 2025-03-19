import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import string
import ref_dialogue_capture


nltk.download('punkt')
nltk.download('wordnet')


def preprocess(text):
    # lowercase and remove punctuation
    tokens = nltk.word_tokenize(text.lower())
    return [t for t in tokens if t not in string.punctuation]


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
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score


def evaluate_corpus(references_list, candidates_list):
    smoothie = SmoothingFunction().method1

    # preprocess references and candidates
    references_tokens = [[preprocess(ref)] for ref in references_list]
    candidates_tokens = [preprocess(cand) for cand in candidates_list]

    # corpus-level BLEU
    bleu1 = corpus_bleu(references_tokens, candidates_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = corpus_bleu(references_tokens, candidates_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu4 = corpus_bleu(references_tokens, candidates_tokens, smoothing_function=smoothie)

    # corpus-level ROUGE-L and METEOR (average across all samples)
    rouge_scores = []
    meteor_scores = []
    for refs, cand in zip(references_tokens, candidates_tokens):
        rouge = max([rouge_l(cand, ref) for ref in refs])
        rouge_scores.append(rouge)

    meteor_scores = [meteor_score(refs, cand) for refs, cand in zip(references_list, candidates_list)]

    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    return bleu1, bleu2, bleu4, avg_rouge, avg_meteor


# example usage
if __name__ == "__main__":
    # ref text for tokenization in evaluate_corpus
    all_dialogue = '../training_data/training_text/isolated_categories/darcy_dialogue_only.txt'
    references = ref_dialogue_capture.capture_references(all_dialogue, min_words=12, max_words=80, sample_size=100)

    # candidate text generated by the model
    candidates = [
        "I do not like to be the first to break silence with a stranger whom I only saw once before; and if I do speak, I shall probably make my words too stiff, and my manner too stately and distant; so if you will be so good, cousin, as to tell me, first, who are these people with you?--for I did not want to look at the woman, because I was half ashamed of the dress she wore. I had never seen any one dressed so."
    ]

    bleu1, bleu2, bleu4, rouge, meteor = evaluate_corpus(references, candidates)

    print(f"Corpus BLEU-1: {bleu1:.4f}")
    print(f"Corpus BLEU-2: {bleu2:.4f}")
    print(f"Corpus BLEU-4: {bleu4:.4f}")
    print(f"Average ROUGE-L: {rouge:.4f}")
    print(f"Average METEOR: {meteor:.4f}")
