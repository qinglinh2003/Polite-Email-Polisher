import json
import random
import argparse
from infer import polite_rewrite
from cli import load_config

def sample_cases(config_path, data_path, num_cases, seed=42, output_path=None, return_md=False):
    config = load_config(config_path)

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(seed)
    samples = random.sample(data, min(num_cases, len(data)))

    results = []
    for item in samples:
        inp = item["input"]
        ref = item["target"]
        pred = polite_rewrite(inp, config)
        results.append((inp, ref, pred))

    md = ["| Input | Reference Output | Model Prediction |",
          "|------|----------|----------|"]

    for inp, ref, pred in results:
        md.append(f"| `{inp}` | `{ref}` | `{pred}` |")
    table = "\n".join(md)

    if return_md:
        return table

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(table)
        print(f"Case table written to {output_path}")
    else:
        print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--data", required=True, help="JSON data path")
    parser.add_argument("--num_cases", type=int, default=5, help="Number of cases to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", help="Optional path to save Markdown table")
    args = parser.parse_args()

    sample_cases(
        config_path=args.config,
        data_path=args.data,
        num_cases=args.num_cases,
        seed=args.seed,
        output_path=args.output
    )


