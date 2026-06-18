import json
import scipy.stats as stats

def run_analysis():
    with open("phase1_results.json", "r") as f:
        data = json.load(f)

    # Filter out any parse failures
    valid_data = [d for d in data if d["status"] == "Success"]

    monotone_correct = 0
    monotone_total = 0
    non_monotone_correct = 0
    non_monotone_total = 0

    for d in valid_data:
        if d["is_monotone"]:
            monotone_total += 1
            if d["is_correct"]:
                monotone_correct += 1
        else:
            non_monotone_total += 1
            if d["is_correct"]:
                non_monotone_correct += 1

    # Calculate Accuracies
    monotone_acc = (monotone_correct / monotone_total) * 100 if monotone_total > 0 else 0
    non_monotone_acc = (non_monotone_correct / non_monotone_total) * 100 if non_monotone_total > 0 else 0
    accuracy_gap = monotone_acc - non_monotone_acc

    # Fisher's Exact Test
    # [[Monotone Correct, Monotone Incorrect], [Non-Monotone Correct, Non-Monotone Incorrect]]
    contingency_table = [
        [monotone_correct, monotone_total - monotone_correct],
        [non_monotone_correct, non_monotone_total - non_monotone_correct]
    ]
    
    _, p_value = stats.fisher_exact(contingency_table, alternative='greater')

    print("=" * 50)
    print("PHASE 1 REPLICATION RESULTS")
    print("=" * 50)
    print(f"Total Valid Trajectories: {len(valid_data)}")
    print(f"\nMonotone Trajectories: {monotone_total}")
    print(f"  -> Accuracy: {monotone_acc:.2f}% ({monotone_correct}/{monotone_total})")
    
    print(f"\nNon-Monotone Trajectories: {non_monotone_total}")
    print(f"  -> Accuracy: {non_monotone_acc:.2f}% ({non_monotone_correct}/{non_monotone_total})")
    
    print("\n" + "-" * 50)
    print(f"ACCURACY GAP: +{accuracy_gap:.2f} percentage points")
    print(f"P-VALUE: {p_value:.5f}")
    
    if p_value < 0.05:
        print("\n[DECISION GATE PASSED]")
        print("The gap is statistically significant. Entropy trajectory predicts correctness.")
        print("We are cleared to begin Phase 2: Attention Instrumentation.")
    else:
        print("\n[DECISION GATE FAILED]")
        print("The gap is not statistically significant.")
        print("We must escalate to a larger model (e.g., 7B) or adjust our monotonicity threshold.")

if __name__ == "__main__":
    run_analysis()