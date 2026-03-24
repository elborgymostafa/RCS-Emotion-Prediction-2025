import os
import json
import requests
from time import sleep

# -----------------------------
# API Key and Base URL
# -----------------------------
# You can keep this hardcoded for now, or use environment variables.
API_KEY = os.getenv("MISTRAL_API_KEY", "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")
BASE_URL = os.getenv("MISTRAL_BASE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")

# -----------------------------
# Paths
# -----------------------------
IN_PATH = "recent_doc_without_polarity.jsonl"
OUT_PATH = "mistral_zeroshot_updatedprompt.jsonl"

# -----------------------------
# Prompt Template
# -----------------------------
PROMPT_TEMPLATE = """
<s>[INST]
You are an expert **Restaurant Review Emotion Classifier**. 

## INPUT FORMAT You have the following input format {{"input": "{input}", "output": "{output}"}}

## TASK For each aspect and its polarity provided in the input, predict the **single most dominant emotion** expressed about that specific aspect. 

**CRITICAL**: Aspect MUST be copied EXACTLY as given. Do NOT change, infer, or correct aspect. 

## DOMAIN: RESTAURANT REVIEWS Focus only on restaurant dining experiences. 

## 2. ASPECT DEFINITIONS Each review may refer to one or more of the following
aspects: 
- **food**: Taste, freshness, appearance, or quality of dishes, portion (lower priority). Includes: food, drinks, breakfast, lunch, and dinner. 
- **staff**: Behavior, friendliness, professionalism, or attitude of employees, a person is explicitly mentioned. Includes: waiter, manager, busboy , host , waitress.
- **service**: Wait times, speed, order accuracy, payment, and reservations (not staff behavior), table and seat management. 
- **price**: Perceived cost, fairness, or value for money. 
- **ambience**: Sensory atmosphere - lighting, decor, noise, temperature, smell, cleanliness. 
- **place**: Includes physical features both outside (geographical location, accessibility, or parking) and inside (tables, space, bar, and other parts inside the restaurant) 
- **menu**: Variety of food and drinks, dietary options, readability, and item availability. 
- **miscellaneous**: Overall experience or the category that does not fit any other aspect. (e.g "Reserve a cozy window seat for more privacy, or hop onto a stool at the bar to dine solo")

## EMOTION DECISION PROCESS 

## EMOTION DEFINITIONS & WHEN TO USE EACH 

- **admiration** → Strong praise, can be highly descriptive (e.g. "best ever", "outstanding", "perfect", "amazing")
- **satisfaction** → Content approval without excitement, meets expectations, feeling relaxed and comfortable (e.g. "pleasant", "nice", "enjoyable", "satisfying", "good", "happy")
- **gratitude** → ONLY for the aspects "staff" and "service": Thankfulness for effort/help/accommodation (e.g "thank you", "grateful", "helpful", "she helped us").
- **disgust** → ONLY for the aspects "food" and "place": Sensory aversion; strong 'gross' reaction (e.g. "rotten", "unsafe", "greasy", "nasty", "dirty", "filthy area")
- **disappointment** → Deep and strong negative emotional response. It typically arises when expectations are not met and often concerns multiple qualities or the overall performance of an aspect (unlike annoyance) (e.g. "extremely bad", "the worst", "expected better", "disappointing",”let down”)
- **annoyance** → Reflects a milder, more situational negative reaction. It usually refers to one specific issue, incident, or short-lived situation within an aspect. The attitude is irritated or frustrated, but not strongly negative and not focused on the aspect. (e.g "annoying")
- **neutral** → mild emotion without explicit positive or negative reaction (e.g "average", "normal", "ok", "nothing special")
- **mentioned_only** → Aspect referenced or factually described but no opinion expressed (e.g. "I ate the pasta" )
- **mixed_emotions** → CLEAR opposing feelings about SAME aspect (e.g "the pizza was very good, but the pasta was too salty")

## DECISION WORKFLOW 

For EACH aspect in input: 
1. Choose one polarity out of these three (positive/negative/neutral) 
2. Choose corresponding ONLY ONE emotion from the provided list 

## OUTPUT FORMAT 
Return EXACTLY this structure
{{
  "input":{input},
  "output": [
    {{"aspect": "...", “polarity”: "...","emotion": "..."}},
    {{"aspect": "...", “polarity”: "...","emotion": "..."}},
    ...
  ]
}}
-Keep "input" with the full review text
-Keep "output" as an array of objects with each object having aspect(unchanged from input),polarity(unchanged from input) and emotion(your prediction)
Output ONLY valid JSON objects, one per line. No explanations, no markdown, no extra text. 

[/INST]
"""

# -----------------------------
# Query Function
# -----------------------------
def query_mistral(prompt: str, model="mistralai/mistral-large-3-675b-instruct-2512", max_retries=3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert aspect-based emotion analysis model. Respond only with valid JSON, without ``` or code fences."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.15,
        "top_p": 1.0,
        "max_tokens": 128,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(BASE_URL, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Attempt {attempt+1} failed: {e}")
            sleep(3)
    return None


# -----------------------------
# Process All Reviews
# -----------------------------
with open(IN_PATH, "r", encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin, 1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
            review = obj.get("input", "")
            outputs = obj.get("output", [])
            output_block = ", ".join(
            [f'{{"aspect": "{o["aspect"]}", "polarity":null, "emotion": null}}'
            for o in outputs]
)
        except json.JSONDecodeError as e:
            print(f"Skipping line {i}: Invalid JSON ({e})")
            continue
        prompt = PROMPT_TEMPLATE.format(input=review,output=output_block)
        print(f"Processing line {i}...")

        output_text = query_mistral(prompt)
        if not output_text:
            print(f"No response for line {i}")
            continue

        # Try to parse model response as JSON
        try:
            parsed_output = json.loads(output_text)
        except json.JSONDecodeError:
            # If model output is not valid JSON, store raw text for debugging
            print(f"⚠️ Line {i}: Model returned non-JSON output.")
            parsed_output = {"raw_response": output_text}

        fout.write(json.dumps(parsed_output, ensure_ascii=False) + "\n")
        fout.flush()

print(f"Done. Results saved to {OUT_PATH}")
