
import json
from pathlib import Path

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

BASE_DIR = Path(__file__).resolve().parent

# Step 1: Load saved evaluation data
with open(BASE_DIR / "final_output.json", encoding="utf-8") as f:
    eval_data = json.load(f)

# Step 2: Connect to the locally running Ollama server
client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

# Use the model installed on your machine
evaluator_llm = llm_factory(
    "gemma3:4b",
    provider="openai",
    client=client,
    max_tokens=4096,
)

# Step 3: Initialize faithfulness evaluator
faithfulness = Faithfulness(llm=evaluator_llm)

# Step 4: Evaluate each saved question
scores = []

for item in eval_data:
    result = faithfulness.score(
        user_input=item["question"],
        response=item["answer"],
        retrieved_contexts=item["context"]
    )

    print("\nQuestion:", item["question"])
    print("Faithfulness score:", result.value)

    scores.append(result.value)

# Step 5: Calculate average faithfulness
if scores:
    print("\nAverage Faithfulness:", sum(scores) / len(scores))
