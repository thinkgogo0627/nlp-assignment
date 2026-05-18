"""Deep peek at multi-agent pipeline state.

Shows for each example:
  - question / gold
  - decomposition (subquestions + pattern)
  - retrieved evidence count
  - intermediate answers
  - preliminary answer
  - reflector flags
  - final answer
  - raw agent outputs (truncated)

Usage: python peek_multi.py [path] [example_idx]
"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/pilot_multi.json"
specific_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None

data = json.load(open(path))
print(f"=== {path} ===")
print(f"Mode: {data['mode']} | N: {data['num_examples']} | "
      f"EM: {data['em']:.3f} | F1: {data['f1']:.3f}\n")

results = data["results"]
if specific_idx is not None:
    results = [results[specific_idx]]

for r in results:
    print("=" * 90)
    print(f"[idx={r['idx']}] Q: {r['question']}")
    print(f"  Gold: {r['gold_answer']!r}")
    print(f"  Pred: {r['prediction']!r}")
    print(f"  EM: {r['em']:.2f}  F1: {r['f1']:.2f}")

    d = r.get("details", {})
    print(f"\n  --- DECOMPOSITION ---")
    print(f"  pattern: {d.get('reasoning_pattern')}")
    for i, sq in enumerate(d.get("decomposition", []), 1):
        print(f"    {i}. {sq!r}")

    print(f"\n  --- EVIDENCE (count={len(d.get('retrieved_evidence', []))}) ---")
    for i, ev in enumerate(d.get("retrieved_evidence", [])[:4], 1):
        obs = str(ev.get("observation", ""))[:150]
        print(f"    [{i}] Q='{ev.get('query', '')}' → {obs}")

    print(f"\n  --- INTERMEDIATE ANSWERS ---")
    for i, ia in enumerate(d.get("intermediate_answers", []), 1):
        print(f"    sub_q_{i}: {ia.get('subquestion', '')[:60]}")
        print(f"      → answer: {ia.get('answer')!r}")
        print(f"      → quote:  {str(ia.get('evidence_quote', ''))[:80]!r}")

    print(f"\n  --- PRELIMINARY / REFLECTOR ---")
    print(f"  preliminary_answer: {d.get('preliminary_answer')!r}")
    print(f"  reflector_flags: {d.get('reflector_flags')}")

    print(f"\n  --- RAW OUTPUTS (truncated) ---")
    raw = d.get("raw_outputs", {})
    for role in ["decomposer", "actor_query", "actor_answer", "reflector"]:
        val = raw.get(role)
        if isinstance(val, list):
            for i, v in enumerate(val):
                preview = str(v).replace("\n", " ")[:200]
                print(f"  {role}[{i}]: {preview}")
        elif val:
            preview = str(val).replace("\n", " ")[:200]
            print(f"  {role}: {preview}")
        else:
            print(f"  {role}: (none)")
    print()
