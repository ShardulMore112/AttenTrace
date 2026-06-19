import os

def tally_failures(filepath="alignment_failures_debug.txt"):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}. (If you had 0 failures, this is a miracle!)")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    no_regex = text.count("reason=NO_REGEX_MATCH")
    len_mismatch = text.count("reason=LENGTH_MISMATCH")
    total = no_regex + len_mismatch

    print(f"--- FAILURE LOG TALLY ---")
    print(f"Total Trajectories Dropped: {total} / 500")
    if total > 0:
        print(f"NO_REGEX_MATCH (Instruction Amnesia): {no_regex} ({(no_regex/total)*100:.1f}%)")
        print(f"LENGTH_MISMATCH (Tokenization Bug): {len_mismatch} ({(len_mismatch/total)*100:.1f}%)")

tally_failures("alignment_failures_debug.txt")