from hotpotqa_eval import create_dataset, evaluate_single

dataset = create_dataset(n=5)
prompt = (
    "Answer the following question using the provided context. "
    "The answer requires information from multiple paragraphs.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Provide a clear, concise answer based on the information in the context."
)

API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = "gemini-2.5-flash-lite"

total_reward = 0
for i, task in enumerate(dataset):
    print("=" * 80)
    print(f"Task {i+1}/{len(dataset)}")
    print("=" * 80)
    r = evaluate_single(prompt=prompt, task=task, api_base=API_BASE, model=MODEL)

    print(f"Reward: {r['reward']}")
    print()
    print("Feedback:")
    print("-" * 80)
    print(r["feedback"])
    print("-" * 80)
    total_reward += r["reward"]
    print()

print("=" * 80)
print(f"Total: {total_reward}/{len(dataset)} correct")
print("=" * 80)
