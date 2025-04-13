# DarcyGPT

This repository contains training scripts, prompting tools, and evaluation utilities for fine-tuning language models (e.g., GPT-2, GPT-J) on *Pride and Prejudice* dialogue — specifically, Mr. Darcy’s lines.

The project includes:

- Model training pipelines
- Prompting interface (interactive and batch)
- Evaluation via BLEU, ROUGE, METEOR, and perplexity
- Graph generation for comparison between models

---

## Folder Setup

Before using the CLI or running evaluation, create the required folders manually (they are `.gitignore`d and not included by default).

Run the following commands in your terminal after navigating to the project folder:

**macOS / Linux**

```
mkdir -p   src/evaluation_data/prompt_sets   src/evaluation_data/prompt_results/compiled_analysis   src/evaluation_data/prompt_results/json   src/evaluation_data/prompt_results/txt   models
```

**Windows (PowerShell)**

(run these commands one at a time)

```
mkdir src\evaluation_data\prompt_sets
mkdir src\evaluation_data\prompt_results\compiled_analysis
mkdir src\evaluation_data\prompt_results\json
mkdir src\evaluation_data\prompt_results\txt
mkdir models
```

> Tip: If you're using Git Bash or WSL on Windows, the macOS/Linux version will also work.

---

## Running Commands

All scripts in this project can be run via a single CLI entry point:

```bash
python cli.py <command> [optional args]
```

---

## Model Training Commands

WARNING: depending on your available computing, running these all at once may take a _very_ long time.

```bash
python cli.py train-gpt2medium-1
python cli.py train-gpt2medium-2
python cli.py train-gpt2large-1
python cli.py train-gpt2large-2
python cli.py train-gptj-1
python cli.py train-gptj-2
```

> These scripts do not take additional arguments. Models are saved to the `models/` folder automatically. If you'd like to edit the training arguments, this may be done by editing the individual files.

---

## Prompting Models

Run the prompting interface:

```bash
python cli.py prompt
```

You will be prompted to:
- Choose interactive mode (enter a prompt yourself), **or**
- Use file mode (automated generation from a list of prompts)

---

### File Mode: Setup Instructions

1. Add your prompts in a plain text file, one prompt per line.  
   Name it like: `prompt_set_1.txt`.

2. Place the file in:
   ```
   src/evaluation_data/prompt_sets/
   ```

3. Open `prompt.py` and set:
   ```python
   set_num = 1  # <- change this to match your file name
   ```

4. Optionally: comment out any models you **don't** want to run at the top of `prompt.py`.

5. Then run:
   ```bash
   python cli.py prompt
   ```

6. The output will be saved automatically to:
   ```
   src/evaluation_data/prompt_results/txt/prompt_results_1.txt
   ```

---

## Evaluation & Visualization

Once your results are generated, you can run evaluation and graphing tools.

### Convert `.txt` results to JSON

```bash
python cli.py convert-results-json 1
```
- `1` refers to `set_num` used during prompting.

---

### Run Evaluation Metrics

```bash
python cli.py evaluate-metrics
```

---

### Prepare & Visualize Results

```bash
python cli.py assemble-results
python cli.py prep-for-graphing
python cli.py collect-best-results
python cli.py graph-vitals
python cli.py graph-all
```

---

## Dataset Builders

Generate training data using:

```bash
python cli.py build-dataset-1
python cli.py build-dataset-2
```

---

## Notes

- Models that have not been trained will fail when selected for prompting — make sure to train only the models you intend to use.
- Evaluation folders are ignored in version control, so be sure to create them before running prompt or evaluation scripts.
- All scripts assume you are running them from the top-level project directory.

---

## Dev Setup

This project uses plain Python scripts and relies on the following libraries:

```
pip install transformers datasets evaluate torch peft nltk scikit-learn matplotlib seaborn numpy
```

Some evaluation scripts use BLEU, ROUGE, and METEOR from `nltk` and `evaluate`.
If you run into issues with NLTK, you may also need to run:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

Make sure you're using Python 3.8+.
```bash
pip install transformers datasets evaluate
```

---

This project is a personal research and experimentation effort.

Users are welcome to explore or adapt at their leisure.
