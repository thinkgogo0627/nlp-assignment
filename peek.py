"""Quick peek at pilot/baseline predictions. Usage: python peek.py [path]"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/pilot_single.json"
data = json.load(open(path))

print(f"=== {path} ===")
print(f"Mode: {data['mode']} | N: {data['num_examples']} | "
      f"EM: {data['em']:.3f} | F1: {data['f1']:.3f}\n")

for r in data["results"]:
    mark = "O" if r["em"] >= 0.99 else "X"
    print(f"[{mark}] Q: {r['question']}")
    print(f"    Pred: {r['prediction']!r}")
    print(f"    Gold: {r['gold_answer']!r}")
    print(f"    EM:{r['em']:.2f}  F1:{r['f1']:.2f}")
    print()
