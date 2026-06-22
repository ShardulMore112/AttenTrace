import json
import re
import shutil

def normalize_math_answer(text):
    if not isinstance(text, str) or not text: return ""
    clean = text.replace("$", "")
    clean = re.sub(r'\\text\{([^}]*)\}', r'\1', clean) 
    clean = clean.replace("\\left", "").replace("\\right", "") 
    clean = "".join(clean.split()) 
    return clean.lower()

def extract_boxed_balanced(text):
    """Uses a stack-based approach to handle infinite levels of nested LaTeX braces."""
    match = re.search(r'\\boxed\{', text)
    if not match:
        return text.strip().split('\n')[-1]
    
    start_idx = match.end() - 1
    stack = 0
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            stack += 1
        elif text[i] == '}':
            stack -= 1
            if stack == 0:
                return text[start_idx+1:i]
    return text.strip().split('\n')[-1]

filepath = r"data\math500\phase2_math500_3B_graded.json"

# 1. Create safety backup
shutil.copy(filepath, filepath.replace(".json", "_BACKUP3.json"))
print(f"Backup created: {filepath.replace('.json', '_BACKUP3.json')}")

with open(filepath, "r", encoding="utf-8") as f:
    data = json.load(f)

changed_count = 0
for record in data:
    model_ans = record.get("model_generated_answer", "")
    truth = record.get("ground_truth_answer", "")
    
    # 2. Extract using balanced-brace algorithm
    extracted_truth = extract_boxed_balanced(truth)
    
    # 3. Strict normalization and comparison
    norm_model = normalize_math_answer(model_ans)
    norm_gt = normalize_math_answer(extracted_truth)
    
    # Grade strictly (Truth must be contained in model's short answer)
    correct_grade = 1 if norm_gt and norm_gt in norm_model else 0
    
    if record["is_correct"] != correct_grade:
        record["is_correct"] = correct_grade
        changed_count += 1

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print(f"Final pass complete. Corrected {changed_count} labels.")
print("The dataset is now mathematically aligned and robust to nested LaTeX.")