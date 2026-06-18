import torch
import json
import re
import gc
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import kendalltau

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "gsm8k_subset.json"
DTYPE = torch.bfloat16
DEVICE = "cuda"

def load_environment():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation="eager"  # Explicitly required to materialize attention matrices
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def calculate_token_entropies(logits):
    probs = torch.softmax(logits, dim=-1)
    epsilon = 1e-9
    return -torch.sum(probs * torch.log2(probs + epsilon), dim=-1)

def align_steps_to_tokens(generated_text, full_token_ids, tokenizer, prompt_len):
    encoding = tokenizer(generated_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding.offset_mapping
    
    generated_ids = full_token_ids[prompt_len:]
    if len(generated_ids) - len(offsets) == 1:
        generated_ids = generated_ids[:-1]
    elif len(offsets) != len(generated_ids):
        return None, None
        
    step_pattern = re.compile(r"(?m)^(Step \d+:.*?)(?=^Step \d+:|^Final Answer:|\Z)", re.DOTALL)
    step_spans = []
    
    for match in step_pattern.finditer(generated_text):
        char_start, char_end = match.span(1)
        start_token_idx = None
        end_token_idx = None
        
        for idx, (tok_char_start, tok_char_end) in enumerate(offsets):
            if start_token_idx is None and tok_char_start <= char_start < tok_char_end:
                start_token_idx = idx
            if tok_char_end <= char_end:
                end_token_idx = idx
                
        if start_token_idx is not None and end_token_idx is not None:
            step_spans.append((start_token_idx + prompt_len, end_token_idx + 1 + prompt_len))
            
    return step_spans, generated_ids

def plot_trajectories(out_entropies, attn_entropies, attn_norm, problem_id):
    steps = list(range(1, len(out_entropies) + 1))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:red'
    ax1.set_xlabel('Reasoning Step')
    ax1.set_ylabel('Output Entropy', color=color1)
    ax1.plot(steps, out_entropies, marker='o', color=color1, label='Output')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    color3 = 'tab:green'
    ax2.set_ylabel('Attention Entropy', color='black')
    ax2.plot(steps, attn_entropies, marker='s', linestyle='--', color=color2, label='Attn (Raw)')
    ax2.plot(steps, attn_norm, marker='^', color=color3, label='Attn (Norm)')
    ax2.tick_params(axis='y', labelcolor='black')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    plt.title(f'Entropy Trajectories (GSM8K ID: {problem_id})')
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"trajectory_pilot_{problem_id}.png", dpi=300)
    print(f"\n[PLOT SAVED] trajectory_pilot_{problem_id}.png generated.")

def process_pilot(model, tokenizer, dataset):
    # Initialize output file if it doesn't exist
    if not os.path.exists("pilot_results.json"):
        with open("pilot_results.json", "w", encoding="utf-8") as f:
            json.dump([], f)

    for i, problem in enumerate(dataset):
        print(f"\n{'='*50}\n--- Processing Problem {problem['id']} ({i+1}/{len(dataset)}) ---\n{'='*50}")
        
        # Check if problem was already processed in a prior run
        try:
            with open("pilot_results.json", "r", encoding="utf-8") as f:
                existing_records = json.load(f)
            if any(record["id"] == problem["id"] for record in existing_records):
                print(f"Problem {problem['id']} already found in JSON. Skipping to next.")
                continue
        except (FileNotFoundError, json.JSONDecodeError):
            existing_records = []

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
        prompt_len = inputs.input_ids.shape[1]
        
        # --- PASS 1: GENERATION ---
        with torch.no_grad():
            outputs_gen = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
            
        full_seq = outputs_gen.sequences[0]
        gen_ids = full_seq[prompt_len:]
        logits = torch.cat(outputs_gen.scores, dim=0)
        
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        step_spans, clean_gen_ids = align_steps_to_tokens(generated_text, full_seq, tokenizer, prompt_len)
        
        if not step_spans:
            print("[DEBUG] Alignment mapping failed. Skipping problem.")
            continue
            
        # --- STRICT BPE EXACT-MATCH GATE ---
        reencoded_ids = tokenizer(generated_text, add_special_tokens=False).input_ids
        if reencoded_ids != clean_gen_ids.tolist():
            print(f"[DEBUG] BPE Mismatch! Re-encoded: {len(reencoded_ids)}, Clean original: {len(clean_gen_ids)}. Skipping problem.")
            continue
            
        if len(logits) - len(clean_gen_ids) == 1:
            logits = logits[:-1]
            
        out_token_entropies = calculate_token_entropies(logits)
        
        del outputs_gen
        torch.cuda.empty_cache()
        gc.collect()

        # --- PASS 2: ATTENTION EXTRACTION ---
        print("Extracting full attention matrices...")
        pass2_inputs = full_seq.unsqueeze(0) 
        seq_len = pass2_inputs.shape[1]
        
        with torch.no_grad():
            outputs_extract = model(
                pass2_inputs,
                output_attentions=True,
                use_cache=False 
            )
            
        attentions = outputs_extract.attentions
        
        if i == 0:
            print(f"\n[CHECK 1 & 2] Tensor Shapes")
            print(f"Total Layers: {len(attentions)}")
            print(f"Layer 0 Shape: {attentions[0].shape}")
            print(f"Sequence Length: {seq_len}")
            
            allocated_vram = torch.cuda.memory_allocated() / (1024**3)
            print(f"\n[CHECK 4] VRAM Allocated: {allocated_vram:.2f} GB")
            
            print("\n[CHECK 3] Attention Row Sums (Sample layers)")
            for l_idx in [0, 14, 27]:
                row_sums = attentions[l_idx][0].sum(dim=-1)
                print(f"Layer {l_idx:02d} | Min: {row_sums.min().item():.4f}, Max: {row_sums.max().item():.4f}")
            
        epsilon = 1e-9
        layer_entropies = []
        
        n_attended_tokens = torch.arange(1, seq_len + 1, device=DEVICE, dtype=DTYPE)
        max_entropy_bounds = torch.log2(n_attended_tokens)
        max_entropy_bounds = torch.clamp(max_entropy_bounds, min=1e-9) 
        
        for l in range(len(attentions)):
            attn_probs = attentions[l][0] 
            head_entropies = -torch.sum(attn_probs * torch.log2(attn_probs + epsilon), dim=-1) 
            mean_head_entropy = head_entropies.mean(dim=0)
            layer_entropies.append(mean_head_entropy)
            
        global_attn_entropy = torch.stack(layer_entropies).mean(dim=0)
        global_attn_entropy_norm = global_attn_entropy / max_entropy_bounds
        
        # --- STEP AGGREGATION ---
        step_out_entropies = []
        step_attn_global = []
        step_attn_global_norm = []
        step_attn_std = []
        step_token_lengths = []
        step_layer_dict = {f"layer_{l}": [] for l in range(len(attentions))}
        
        for start_idx, end_idx in step_spans:
            step_length = end_idx - start_idx
            step_token_lengths.append(step_length)
            
            out_span = out_token_entropies[start_idx - prompt_len : end_idx - prompt_len]
            step_out_entropies.append(out_span.mean().item())
            
            attn_span = global_attn_entropy[start_idx:end_idx]
            step_attn_global.append(attn_span.mean().item())
            step_attn_std.append(attn_span.std().item() if len(attn_span) > 1 else 0.0)
            
            attn_span_norm = global_attn_entropy_norm[start_idx:end_idx]
            step_attn_global_norm.append(attn_span_norm.mean().item())
            
            for l in range(len(attentions)):
                layer_span = layer_entropies[l][start_idx:end_idx]
                step_layer_dict[f"layer_{l}"].append(layer_span.mean().item())
                
        if i == 0:
            plot_trajectories(step_out_entropies, step_attn_global, step_attn_global_norm, problem["id"])
            
        out_tau, _ = kendalltau(list(range(len(step_out_entropies))), step_out_entropies)
        attn_tau, _ = kendalltau(list(range(len(step_attn_global_norm))), step_attn_global_norm)
        
        # --- PROGRESSIVE SAVE FILE UPDATE ---
        new_record = {
            "id": problem["id"],
            "total_seq_len": seq_len,
            "step_token_lengths": step_token_lengths,
            "out_tau": out_tau if not np.isnan(out_tau) else 0.0,
            "attn_tau": attn_tau if not np.isnan(attn_tau) else 0.0,
            "step_out_mean": step_out_entropies,
            "step_attn_global": step_attn_global,
            "step_attn_global_norm": step_attn_global_norm,
            "step_attn_std": step_attn_std,
            "layer_wise_attn": step_layer_dict
        }
        
        # Re-read to guarantee sync with disk across multiple runs/processes
        try:
            with open("pilot_results.json", "r", encoding="utf-8") as f:
                current_records = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            current_records = []
            
        current_records.append(new_record)
        
        with open("pilot_results.json", "w", encoding="utf-8") as f:
            json.dump(current_records, f, indent=4)
            
        print(f"[PROGRESS LOCKED] Problem {problem['id']} successfully appended to pilot_results.json")
        
        del outputs_extract
        del attentions
        torch.cuda.empty_cache()
        gc.collect()

    print("\nExtraction script completely finished execution loop.")

if __name__ == "__main__":
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    model, tokenizer = load_environment()
    process_pilot(model, tokenizer, dataset)