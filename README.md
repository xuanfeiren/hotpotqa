# HotpotQA Prompt Evaluation

## Dataset

HotpotQA is a multi-hop question answering dataset. Each task requires reasoning across multiple paragraphs to answer a question.

- **Source**: HuggingFace `hotpot_qa`, `distractor` configuration
- **Split**: `validation` (7,405 tasks)
- **Structure**: Each task contains:
  - `question`: A question requiring multi-hop reasoning
  - `context`: 10 paragraphs (2-3 relevant, rest are distractors), each with a title and sentences
  - `answer`: A short ground-truth answer (typically a few words)

Example:
```
Question: Were Scott Derrickson and Ed Wood of the same nationality?
Context:  10 paragraphs about Scott Derrickson, Ed Wood, and unrelated topics
Answer:   yes
```

## Installation

```bash
pip install datasets openai tqdm
```

Set your API key:
```bash
export GEMINI_API_KEY="your_key"
# or
export OPENAI_API_KEY="your_key"
```

## API

### `create_dataset(n=20) -> List[Task]`

Loads the first `n` tasks from the HotpotQA validation set. Each `Task` has three fields:

- `task.question` — the question string
- `task.context` — pre-formatted paragraphs (10 paragraphs with titles, joined by newlines)
- `task.answer` — the ground-truth answer string

The dataset is downloaded from HuggingFace on first call and cached locally afterwards.

```python
from hotpotqa_eval import create_dataset

dataset = create_dataset(n=100)
print(dataset[0].question)  # "Were Scott Derrickson and Ed Wood of the same nationality?"
print(dataset[0].answer)    # "yes"
print(dataset[0].context)   # "Paragraph 1 (Ed Wood (film)):\n..."
```

### `evaluate_single(prompt, task, api_base, model) -> dict`

Evaluates a prompt template on a **single** task. Returns a binary reward and a structured text feedback.

- `prompt`: A string with `{context}` and `{question}` placeholders
- `task`: A single `Task` object
- `api_base`: OpenAI-compatible API base URL
- `model`: Model name

Returns:
```python
{
    "reward": 1 or 0,       # binary correctness
    "feedback": "..."        # structured text (see below)
}
```

The `feedback` string contains:
```
[Question]
<the question>

[Context]
<all 10 paragraphs>

[Expected Answer]
<ground-truth answer>

[Model Output]
<what the model actually responded>

[Correct]
Yes / No
```

Matching logic: case-insensitive, strips trailing punctuation, counts as correct if output equals the answer OR the answer is a substring of the output.

Example:
```python
from hotpotqa_eval import create_dataset, evaluate_single

dataset = create_dataset(n=5)
prompt = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely:"

r = evaluate_single(
    prompt=prompt,
    task=dataset[0],
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    model="gemini-2.5-flash-lite",
)
print(r["reward"])    # 1
print(r["feedback"])  # full structured text
```

### `evaluate(prompt, dataset, api_base, model) -> dict`

Evaluates a prompt template on a **list** of tasks. Returns aggregate accuracy and per-task details.

Returns:
```python
{
    "accuracy": 0.76,            # float 0.0-1.0
    "correct": 76,
    "total": 100,
    "details": [                 # per-task results
        {"question": "...", "expected": "...", "output": "...", "correct": True},
        ...
    ]
}
```

Example:
```python
from hotpotqa_eval import create_dataset, evaluate

dataset = create_dataset(n=100)
prompt = (
    "Answer the following question using the provided context. "
    "The answer requires information from multiple paragraphs.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\n"
    "Provide a clear, concise answer based on the information in the context."
)

result = evaluate(
    prompt=prompt,
    dataset=dataset,
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    model="gemini-2.5-flash-lite",
)
print(f"Accuracy: {result['accuracy']:.1%}")  # Accuracy: 75.3%
```

## Baseline Results

Using `gemini-2.5-flash-lite` with the default prompt:

| Samples | Accuracy |
|---------|----------|
| 100     | 76.0%    |
| 300     | 75.3%    |

## Files

- `hotpotqa_eval.py` — the evaluation library (`create_dataset`, `evaluate_single`, `evaluate`)
- `demo.py` — runnable demo that evaluates the baseline prompt on 5 tasks
