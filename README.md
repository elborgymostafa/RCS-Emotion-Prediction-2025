# RCS Emotion Prediction 2025

A collaborative research project for **emotion-aware Aspect-Based Sentiment Analysis (ABSA)** on restaurant reviews. The goal is to predict fine-grained emotions tied to specific aspects of a review (e.g., food, service, ambiance) using the **MAMS dataset**.

Each of the 6 contributors works on their own branch and explores different approaches — from classical ML (Logistic Regression, SVM) to fine-tuned transformers (BERT, RoBERTa) to LLM prompting (Gemini, GPT-4).

---

## Contributors & Branches

| Branch | Contributor |
|---|---|
| `dev/daniel` | Daniel |
| `dev/Danila` | Danila |
| `dev/julia22-lav` | Julia |
| `dev/sama` | Sama |
| `dev/mustafa` | Mustafa |
| `dev/mane` | Mane |

---

## Project Structure

```
RCS-Emotion-Prediction-2025/
├── data/               # MAMS-ACSA and MAMS-ATSA datasets (do not modify)
├── src/<your_name>/    # Your code goes here
├── results/<your_name> # Your outputs/evaluations go here
├── prompts/<your_name> # Your prompts go here
├── tool/util.py        # Shared utilities
├── paper/              # LaTeX paper
└── requirements.txt    # Python dependencies
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy .env and fill in your API keys (Gemini, OpenAI, HuggingFace)
```

---

## Guidelines

- **Work on your own branch** — never commit directly to `main`
- **Never modify the original datasets** in `data/`
- **No absolute paths** — use relative paths so code works on any machine
- Store code in `src/your_name/`, results in `results/your_name/`, prompts in `prompts/your_name/`
- Prefer Jupyter notebooks for readability, but `.py` files are fine too
- `tool/util.py` is in `.gitignore` — define your own path constants there if needed
- Write modular code
