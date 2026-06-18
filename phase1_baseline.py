import torch
import json
import re
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import kendalltau

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "gsm8k_subset.json"
RESULTS_PATH = "phase1_results.json"
DTYPE = torch.bfloat16
DEVICE = "cuda"

def load_environment():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        device_map=DEVICE
    )
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def calculate_token_entropies(logits):
    """
    Computes Shannon entropy across the full vocabulary for each token.
    logits shape: (sequence_length, vocab_size)
    """
    # Full vocabulary softmax (no temperature scaling, no truncation)
    probs = torch.softmax(logits, dim=-1)
    
    # Shannon entropy: H(x) = -sum(P(x) * log2(P(x) + epsilon))
    epsilon = 1e-9
    entropies = -torch.sum(probs * torch.log2(probs + epsilon), dim=-1)
    
    return entropies

def align_steps_to_tokens(generated_text, generated_token_ids, tokenizer):
    """
    Uses character offset mapping to flawlessly align regex string spans 
    back to the original generation token indices.
    """
    encoding = tokenizer(
        generated_text, 
        add_special_tokens=False, 
        return_offsets_mapping=True
    )
    offsets = encoding.offset_mapping
    
    if len(offsets) != len(generated_token_ids):
        return None 
        
    step_pattern = re.compile(r"(?m)^(Step \d+:.*?)(?=^Step \d+:|^Final Answer:|\Z)", re.DOTALL)
    step_spans = []
    
    for match in step_pattern.finditer(generated_text):
        char_start, char_end = match.span(1)
        start_token_idx = None
        end_token_idx = None
        
        for idx, (tok_char_start, tok_char_end) in enumerate(offsets):
            if start_token_idx is None and tok_char_start >= char_start:
                start_token_idx = idx
            if tok_char_end <= char_end:
                end_token_idx = idx
                
        if start_token_idx is not None and end_token_idx is not None:
            step_spans.append((start_token_idx, end_token_idx + 1))
            
    return step_spans

def process_problem(problem, model, tokenizer):
    system_prompt = (
        "You are a strict mathematical reasoning engine. You must solve the problem step-by-step. "
        "Every single logical step MUST begin on a new line with 'Step N:'. "
        "When you finish reasoning, you MUST end your response with the exact format 'Final Answer: [number]' and nothing else."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem["question"]}
    ]
    
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text_prompt], return_tensors="pt").to(DEVICE)
    input_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
        
    generated_ids = outputs.sequences[0][input_len:]
    logits = torch.cat(outputs.scores, dim=0) 
    
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # --- ANSWER EXTRACTION ---
    final_answer_match = re.search(r"(?i)final answer.*?\b(\d+)\b", generated_text)
    if not final_answer_match:
        fallback_match = re.findall(r"\b\d+\b", generated_text)
        extracted_answer = fallback_match[-1] if fallback_match else None
    else:
        extracted_answer = final_answer_match.group(1)
        
    is_correct = None
    gt_number = None
    if extracted_answer:
        raw_gt = problem["answer"].split("####")[-1].replace(",", "")
        gt_match = re.findall(r"\b\d+\b", raw_gt)
        if gt_match:
            gt_number = gt_match[-1]
            is_correct = (extracted_answer == gt_number)
            
    if is_correct is None:
        print(f"\n[DEBUG ID {problem['id']}] FAILED GATE: Answer Extraction. Extracted: {extracted_answer}, GT: {gt_number}")
        return {"status": "ParseFailure"}
            
    # --- ALIGNMENT FIX ---
    encoding = tokenizer(generated_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding.offset_mapping
    
    if len(generated_ids) - len(offsets) == 1:
        generated_ids = generated_ids[:-1]
        logits = logits[:-1]
    elif len(offsets) != len(generated_ids):
        print(f"\n[DEBUG ID {problem['id']}] FAILED GATE: Token Mismatch! Offsets: {len(offsets)}, Original IDs: {len(generated_ids)}")
        return {"status": "ParseFailure"}
        
    # --- METRICS CALCULATION ---
    token_entropies = calculate_token_entropies(logits)
    step_spans = align_steps_to_tokens(generated_text, generated_ids, tokenizer)
    
    if not step_spans or len(step_spans) < 2:
        print(f"\n[DEBUG ID {problem['id']}] FAILED GATE: Step Spans. Found {len(step_spans) if step_spans else 0} steps.")
        return {"status": "ParseFailure"}
        
    step_mean_entropies = []
    step_max_entropies = []
    
    for start_idx, end_idx in step_spans:
        span_entropies = token_entropies[start_idx:end_idx]
        step_mean_entropies.append(span_entropies.mean().item())
        step_max_entropies.append(span_entropies.max().item())
        
    x_indices = list(range(len(step_mean_entropies)))
    tau_mean, _ = kendalltau(x_indices, step_mean_entropies)
    
    if torch.isnan(torch.tensor(tau_mean)):
        tau_mean = 0.0
        
    is_monotone = tau_mean < 0 

    return {
        "status": "Success",
        "id": problem["id"],
        "is_correct": is_correct,
        "is_monotone": bool(is_monotone),
        "tau_score": float(tau_mean),
        "step_mean_entropies": step_mean_entropies,
        "step_max_entropies": step_max_entropies,
        "num_steps": len(step_spans)
    }

def run_phase1():
    model, tokenizer = load_environment()
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    # MICRO-BATCH TEST SLICE
    # dataset = dataset[:10] 
        
    results = []
    parse_failures = 0
    
    print("\nStarting Entropy Trajectory Analysis...")
    for problem in tqdm(dataset, desc="Processing GSM8K"):
        res = process_problem(problem, model, tokenizer)
        results.append(res)
        if res["status"] == "ParseFailure":
            parse_failures += 1
            
    total = len(dataset)
    success_count = total - parse_failures
    yield_rate = (success_count / total) * 100
    
    print("\n" + "="*50)
    print("PHASE 1 EXECUTION COMPLETE")
    print("="*50)
    print(f"Total Problems: {total}")
    print(f"Successful Parses: {success_count}")
    print(f"Parse Failures: {parse_failures}")
    print(f"Yield Rate: {yield_rate:.2f}%")
    
    if yield_rate < 85.0:
        print("\n[WARNING] Yield rate is below 85%. Dataset may be biased.")
    else:
        print("\n[SUCCESS] Yield rate is healthy.")
        
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    run_phase1()