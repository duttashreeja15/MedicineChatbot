
import json
from pathlib import Path

import pandas as pd
from ollama import Client

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "final_output.json"
OUTPUT_FILE = BASE_DIR / "evaluation_results.csv"

client = Client(host="http://localhost:11434")


def evaluate_response(question, answer, context):
    prompt = f"""
You are evaluating a RAG chatbot.

Evaluate the response on two dimensions:

1. Faithfulness:
How well are the factual claims in the answer
supported by the retrieved context?

2. Answer relevance:
How well does the answer address the question?

Give each metric a score from 0 to 1.

Return only a JSON object with:
- faithfulness
- answer_relevance

Question:
{question}

Retrieved context:
{context}

Generated answer:
{answer}
"""

    response = client.chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        options={"temperature": 0}
    )

    scores = json.loads(response.message.content)

    result = {}

    for metric in ("faithfulness", "answer_relevance"):
        value = float(scores[metric])

        if not 0 <= value <= 1:
            raise ValueError(
                f"{metric} must be between 0 and 1"
            )

        result[metric] = value

    return result


def main():
    

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data).fillna("")

    required = {"question", "answer", "context"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain {required}"
        )

    results = []

    for i, row in df.iterrows():
        print(f"Evaluating question {i + 1}/{len(df)}")

        try:
            scores = evaluate_response(
                row["question"],
                row["answer"],
                row["context"]
            )
            results.append(scores)

        except Exception as e:
            print(f"Evaluation failed: {e}")
            results.append({
                "faithfulness": None,
                "answer_relevance": None
            })

    scores_df = pd.DataFrame(results)
    output = pd.concat([df, scores_df], axis=1)

    output.to_csv(OUTPUT_FILE, index=False)

    print("\nEvaluation summary:")
    print(scores_df.mean().round(3))
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
