import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "gsm8k_subset.json"
# We use bfloat16 which is highly optimized for Ampere architectures like your RTX 3050
DTYPE = torch.bfloat16 

def load_model_and_tokenizer():
    print(f"Loading tokenizer and model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Load directly onto the GPU to save system RAM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="cuda"
    )
    print("Model loaded successfully.\n" + "-"*50)
    return model, tokenizer

def generate_and_parse(model, tokenizer, question):
    # STRONGER system prompt to enforce compliance
    system_prompt = (
        "You are a strict mathematical reasoning engine. You must solve the problem step-by-step. "
        "Every single logical step MUST begin on a new line with 'Step N:'. "
        "When you finish reasoning, you MUST end your response with the exact format 'Final Answer: [number]' and nothing else."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text_prompt], return_tensors="pt").to("cuda")
    
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False, # Automatically uses greedy decoding (fixes the warning)
            return_dict_in_generate=True,
            output_scores=True 
        )
    
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # --- FIXED STEP BOUNDARY PARSING ---
    # (?m)^ ensures it ONLY matches "Step N:" if it is at the absolute beginning of a line
    step_pattern = re.compile(r"(?m)^(Step \d+:.*?)(?=^Step \d+:|^Final Answer:|\Z)", re.DOTALL)
    steps = [match.group(1).strip() for match in step_pattern.finditer(generated_text)]
    
    # --- FIXED FINAL ANSWER EXTRACTION ---
    # Looks for "Final Answer" (case insensitive) and grabs the last distinct number
    final_answer_match = re.search(r"(?i)final answer.*?\b(\d+)\b", generated_text)
    final_answer = final_answer_match.group(1) if final_answer_match else "None found"
    
    return generated_text, steps, final_answer, outputs.scores

def run_phase_0_test():
    model, tokenizer = load_model_and_tokenizer()
    
    # Load our local dataset
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    # Grab the first question for our pipeline test
    test_problem = dataset[0]
    
    print(f"QUESTION:\n{test_problem['question']}\n" + "="*50)
    
    full_text, steps, final_answer, scores = generate_and_parse(model, tokenizer, test_problem['question'])
    
    print("\n--- RAW GENERATION ---")
    print(full_text)
    print("\n" + "="*50)
    
    print("\n--- PARSED STEPS ---")
    for i, step in enumerate(steps, 1):
        print(f"Parsed Step {i}:\n  {step}")
        
    print(f"\nExtracted Final Answer: {final_answer}")
    
    # Sanity check the logits tensor
    print("\n--- LOGITS SANITY CHECK ---")
    print(f"Total generated tokens: {len(scores)}")
    print(f"Shape of logits for the first generated token: {scores[0].shape}")
    print("If you see [1, vocab_size] above, the logit capture was successful.")

if __name__ == "__main__":
    run_phase_0_test()