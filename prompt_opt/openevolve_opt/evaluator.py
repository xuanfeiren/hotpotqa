"""
Evaluator for OpenEvolve prompt optimization on HotpotQA.

This evaluator reads an evolved prompt file, formats it with HotpotQA
context/question, calls the LLM, and checks answer correctness.
Uses the first 100 tasks from the HotpotQA validation set.
"""

import os
import sys
import time
import traceback
import yaml

from openai import OpenAI
from datasets import load_dataset, disable_progress_bars
from tqdm import tqdm
import logging

# Suppress verbose HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

disable_progress_bars()

# Read config.yaml for model settings
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

llm_config = config.get("llm", {})
api_base = llm_config.get("api_base", "https://generativelanguage.googleapis.com/v1beta/openai/")
models = llm_config.get("models", [])
TASK_MODEL_NAME = models[0]["name"] if models else "gemini-2.5-flash-lite"
MAX_TOKENS = llm_config.get("max_tokens", 4096)
MAX_RETRIES = config.get("evaluator", {}).get("max_retries", 3)

# Number of samples to use
NUM_SAMPLES = int(os.environ.get("OPENEVOLVE_NUM_SAMPLES", "100"))

# Initialize client
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
client = OpenAI(base_url=api_base, api_key=api_key)
print(f"OpenEvolve Evaluator initialized: model={TASK_MODEL_NAME}, samples={NUM_SAMPLES}")


def load_hotpotqa_dataset():
    """Load HotpotQA validation set."""
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    return ds


def format_context(example):
    """Format context paragraphs from a HotpotQA example."""
    context_parts = []
    for j, (title, sentences) in enumerate(
        zip(example["context"]["title"], example["context"]["sentences"])
    ):
        context_parts.append(f"Paragraph {j+1} ({title}):\n{' '.join(sentences)}")
    return "\n\n".join(context_parts)


def check_answer(output_text, expected):
    """Check if output matches expected answer."""
    if not output_text:
        return False
    output_clean = output_text.strip().lower().rstrip(".,!?;:")
    expected_clean = expected.strip().lower().rstrip(".,!?;:")
    return output_clean == expected_clean or expected_clean in output_clean


def evaluate_prompt_on_dataset(prompt, dataset, num_samples):
    """Evaluate a prompt on a subset of the HotpotQA dataset."""
    correct = 0
    total = 0
    feedback_items = []

    for i in tqdm(range(min(num_samples, len(dataset))), desc="Evaluating"):
        example = dataset[i]
        context = format_context(example)
        question = example["question"]
        expected = example["answer"]

        # Format prompt
        try:
            formatted = prompt.format(context=context, question=question)
        except (KeyError, IndexError) as e:
            print(f"  Warning: Prompt formatting failed for task {i}: {e}")
            total += 1
            feedback_items.append({
                "task_idx": i,
                "correct": False,
                "error": f"Prompt formatting error: {e}",
            })
            continue

        # Call LLM
        output_text = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=TASK_MODEL_NAME,
                    messages=[{"role": "user", "content": formatted}],
                    temperature=0.1,
                    max_tokens=MAX_TOKENS,
                )
                if resp and resp.choices and resp.choices[0].message:
                    output_text = resp.choices[0].message.content
                    if output_text:
                        break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"  Failed task {i} after {MAX_RETRIES} attempts: {e}")
                time.sleep(1)

        is_correct = check_answer(output_text, expected)
        if is_correct:
            correct += 1
        total += 1

        feedback_items.append({
            "task_idx": i,
            "question": question[:100],
            "expected": expected,
            "output": (output_text or "")[:200],
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total, feedback_items


# Cache the dataset
_dataset_cache = None

def _get_dataset():
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = load_hotpotqa_dataset()
    return _dataset_cache


def evaluate_stage1(prompt_path):
    """
    Stage 1: Quick evaluation on 10 samples.
    """
    print("-" * 60)
    print("Stage 1: Quick evaluation (10 samples)")
    print("-" * 60)

    try:
        with open(prompt_path, "r") as f:
            prompt = f.read().strip()

        dataset = _get_dataset()
        accuracy, correct, total, _ = evaluate_prompt_on_dataset(prompt, dataset, num_samples=10)

        print(f"Stage 1 accuracy: {accuracy:.3f} ({correct}/{total})")

        # Features for MAP-Elites
        prompt_length = len(prompt)
        has_cot = any(kw in prompt.lower() for kw in ["step by step", "step-by-step", "reasoning", "think"])
        sophistication = 0.5 if has_cot else 0.2

        return {
            "combined_score": accuracy,
            "prompt_length": prompt_length,
            "reasoning_strategy": sophistication,
        }
    except Exception as e:
        print(f"Stage 1 failed: {e}")
        traceback.print_exc()
        return {"combined_score": 0.0, "prompt_length": 0, "reasoning_strategy": 0.0, "error": str(e)}


def evaluate_stage2(prompt_path):
    """
    Stage 2: Full evaluation on NUM_SAMPLES samples.
    """
    print("-" * 60)
    print(f"Stage 2: Full evaluation ({NUM_SAMPLES} samples)")
    print("-" * 60)

    try:
        with open(prompt_path, "r") as f:
            prompt = f.read().strip()

        dataset = _get_dataset()
        accuracy, correct, total, feedback_items = evaluate_prompt_on_dataset(
            prompt, dataset, num_samples=NUM_SAMPLES
        )

        print(f"Stage 2 accuracy: {accuracy:.3f} ({correct}/{total})")

        # Features for MAP-Elites
        prompt_length = len(prompt)
        has_cot = any(kw in prompt.lower() for kw in ["step by step", "step-by-step", "reasoning", "think"])
        sophistication = 0.5 if has_cot else 0.2

        return {
            "combined_score": accuracy,
            "prompt_length": prompt_length,
            "reasoning_strategy": sophistication,
        }
    except Exception as e:
        print(f"Stage 2 failed: {e}")
        traceback.print_exc()
        return {"combined_score": 0.0, "prompt_length": 0, "reasoning_strategy": 0.0, "error": str(e)}


def evaluate(prompt_path):
    """Main evaluation - delegates to stage2."""
    return evaluate_stage2(prompt_path)
