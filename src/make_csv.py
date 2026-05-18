import json
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--single", required=True)
parser.add_argument("--multi", required=True)
parser.add_argument("--output", default="outputs/results.csv")
args = parser.parse_args()

FILES = {
    "single_agent": args.single,
    "multi_agent": args.multi,
}

rows = []

for setting, path in FILES.items():
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        examples = data.get("results", [])
        split = data.get("task_config", {}).get("dataset", {}).get("split", "dev")
    else:
        examples = data
        split = "dev"

    if not examples:
        raise ValueError(f"No results found in {path}")

    em = sum(float(x["em"]) for x in examples) / len(examples)
    f1 = sum(float(x["f1"]) for x in examples) / len(examples)

    rows.append([
        setting,
        split,
        round(em, 2),
        round(f1, 2),
    ])

with open(args.output, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["setting", "split", "em", "f1"])
    writer.writerows(rows)

print(f"Saved to {args.output}")