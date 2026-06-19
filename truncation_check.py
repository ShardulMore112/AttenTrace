import re

FILE_PATH = "alignment_failures_debug.txt"

with open(FILE_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# Split into individual entries
raw_entries = content.split("=== ")[1:]

entries = []
for raw in raw_entries:
    header, _, body = raw.partition("===\n")
    header = header.strip()
    # header looks like: "math500_1 | reason=NO_REGEX_MATCH"
    parts = header.split("|")
    problem_id = parts[0].strip()
    reason = parts[1].strip().replace("reason=", "") if len(parts) > 1 else "UNKNOWN"
    entries.append({"id": problem_id, "reason": reason, "text": body.strip()})

print(f"Total logged entries: {len(entries)}")

reason_counts = {}
for e in entries:
    reason_counts[e["reason"]] = reason_counts.get(e["reason"], 0) + 1
print("Breakdown by reason:", reason_counts)

# --- Content-based check ---
# We can't measure true final length (snippets are capped at 500 chars by the logger).
# But we CAN check, within that fixed 500-char window, how close the text appears to be
# to finishing: does it contain "Final Answer" or "boxed" anywhere (signal: was close to
# done, got cut right near the finish line) vs. does it cut off mid-equation/mid-sentence
# with no terminal signal at all (signal: still deep in derivation, genuinely needed more room)

near_finish_signals = re.compile(r"final answer|\\boxed|\bthus\b|\btherefore\b.*=", re.IGNORECASE)

# Check whether the snippet ends on a "clean" boundary (full sentence/equation) vs.
# an abrupt mid-token/mid-word/mid-symbol cutoff
def ends_abruptly(text):
    if not text:
        return True
    tail = text.rstrip()
    if not tail:
        return True
    last_char = tail[-1]
    # Clean endings: punctuation, closing brackets/braces from a finished equation
    if last_char in ".!?":
        return False
    return True

abrupt_count = 0
clean_count = 0
near_finish_count = 0
no_near_finish_count = 0

NO_REGEX = [e for e in entries if e["reason"] == "NO_REGEX_MATCH"]

for e in NO_REGEX:
    text = e["text"]
    if ends_abruptly(text):
        abrupt_count += 1
    else:
        clean_count += 1

    if near_finish_signals.search(text):
        near_finish_count += 1
    else:
        no_near_finish_count += 1

n = len(NO_REGEX)
print(f"\n--- Content check on {n} NO_REGEX_MATCH entries ---")
print(f"Snippet ends abruptly (mid-word/mid-equation, no terminal punctuation): {abrupt_count} ({abrupt_count/n*100:.1f}%)")
print(f"Snippet ends on a clean sentence/equation boundary:                     {clean_count} ({clean_count/n*100:.1f}%)")
print(f"Snippet shows 'near finish' language (boxed/therefore/final answer):    {near_finish_count} ({near_finish_count/n*100:.1f}%)")
print(f"Snippet shows NO near-finish language within the 500-char window:       {no_near_finish_count} ({no_near_finish_count/n*100:.1f}%)")

# Rough proxy for "how much work was already done" - count of Step markers visible
# within the truncated window, as an indicator of derivation depth/complexity
step_marker_pattern = re.compile(r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*\*)?Step\s+\d+", re.IGNORECASE)
step_counts = [len(step_marker_pattern.findall(e["text"])) for e in NO_REGEX]
import statistics
print(f"\nStep markers visible within first 500 chars (proxy for derivation complexity):")
print(f"  Mean: {statistics.mean(step_counts):.2f}")
print(f"  Median: {statistics.median(step_counts)}")
print(f"  Max: {max(step_counts)}")
print(f"  Distribution: {sorted(step_counts)[:20]} ... (showing first 20)")

# How many show ZERO step markers within 500 chars (model still in unstructured prose/setup)
zero_steps = sum(1 for c in step_counts if c == 0)
print(f"  Entries with 0 Step markers visible in first 500 chars: {zero_steps} ({zero_steps/n*100:.1f}%)")