import os
import json
from datasets import load_dataset

def prepare_gsm8k_subset(sample_size=150, save_path="gsm8k_subset.json", seed=42):
    print(f"Loading GSM8K dataset from Hugging Face...")
    
    # Load the main split of GSM8K
    dataset = load_dataset("openai/gsm8k", "main")
    test_split = dataset["test"]
    
    print(f"Total problems available in full test split: {len(test_split)}")
    
    # Select a deterministic subset using a fixed seed
    shuffled_test = test_split.shuffle(seed=seed)
    subset = shuffled_test.select(range(min(sample_size, len(shuffled_test))))
    
    # Convert to a list of standard dictionaries
    processed_data = []
    for idx, item in enumerate(subset):
        processed_data.append({
            "id": idx,
            "question": item["question"],
            "answer": item["answer"]
        })
        
    # Save locally to eliminate network overhead in subsequent phases
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully isolated {len(processed_data)} problems.")
    print(f"Subset saved locally to: {os.path.abspath(save_path)}")
    
    # Preview the first item
    print("\n--- Sample Item Preview ---")
    print(f"Question: {processed_data[0]['question']}")
    print(f"Ground Truth Answer Sequence:\n{processed_data[0]['answer']}")
    print("-" * 27)

if __name__ == "__main__":
    # Scoped to 150 problems for rapid iteration during replication checks
    prepare_gsm8k_subset(sample_size=150)