"""
Standardized evaluation atomic unit for HotpotQA.
Ensures that 1 eval call = 1 task evaluation = 1 metric call.
"""

import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any
from datasets import load_dataset, disable_progress_bars
from openai import OpenAI

disable_progress_bars()

@dataclass
class Task:
    question: str
    context: str
    answer: str

def load_hotpotqa_dataset(n: int = 100) -> List[Task]:
    """Load HotpotQA tasks synchronously."""
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    tasks = []
    for i in range(min(n, len(ds))):
        ex = ds[i]
        context_parts = []
        for j, (title, sentences) in enumerate(zip(ex["context"]["title"], ex["context"]["sentences"])):
            context_parts.append(f"Paragraph {j+1} ({title}):\n{' '.join(sentences)}")
        tasks.append(
            Task(
                question=ex["question"],
                context="\n\n".join(context_parts),
                answer=ex["answer"],
            )
        )
    return tasks

def _check_answer(output_text: str, expected: str) -> bool:
    if not output_text:
        return False
    output_clean = output_text.strip().lower().rstrip(".,!?;:")
    expected_clean = expected.strip().lower().rstrip(".,!?;:")
    return output_clean == expected_clean or expected_clean in output_clean

def evaluate_single(
    prompt_template: str,
    task: Task,
    api_base: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Evaluate a single task. This counts as ONE metric call.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    client = OpenAI(base_url=api_base, api_key=api_key)

    formatted = prompt_template.format(context=task.context, question=task.question)
    
    output_text = None
    import random
    
    # Slight initial jitter to avoid 10 threads hitting simultaneously
    time.sleep(random.uniform(0, 0.5))

    for attempt in range(max_retries + 2): # Increased retries for 429s
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": formatted}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if resp and resp.choices and resp.choices[0].message:
                output_text = resp.choices[0].message.content
                if output_text:
                    break
        except Exception as e:
            if "429" in str(e):
                # More aggressive exponential backoff for Flash Lite
                sleep_time = (5 ** attempt) + random.uniform(1, 3)
                time.sleep(sleep_time)
            elif attempt == (max_retries + 1):
                print(f"LLM Call Error: {e}")
            time.sleep(1)

    is_correct = _check_answer(output_text, task.answer)
    
    return {
        "question": task.question,
        "expected": task.answer,
        "output": output_text,
        "correct": is_correct,
    }

# For backward compatibility within projects
def create_dataset(n=20):
    return load_hotpotqa_dataset(n)
