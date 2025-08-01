import json
from tqdm import tqdm
import evaluate
from cli import load_config
from infer import polite_rewrite
from case_sampling import sample_cases 
import os

def evaluate_model(config_path, data_path, num_cases=5, seed=42):
    config = load_config(config_path)
    version = config["model"]["version"]
    exp_dir = os.path.join("experiments", version)
    os.makedirs(exp_dir, exist_ok=True)


    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    references = [item["target"] for item in data]
    predictions = []

    for item in tqdm(data, desc="Evaluating"):
        pred = polite_rewrite(item["input"], config)
        predictions.append(pred)

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")


    bleu_result = bleu.compute(predictions=predictions,
                               references=[[r] for r in references])

    rouge_result = rouge.compute(predictions=predictions, references=references, use_stemmer=True, use_aggregator=True)

    def _get_f1(score):
        return score.mid.fmeasure if hasattr(score, "mid") else float(score)

    metrics = {
        "BLEU": bleu_result["bleu"],
        "ROUGE-1": _get_f1(rouge_result["rouge1"]),
        "ROUGE-L": _get_f1(rouge_result["rougeL"]),
        "date": __import__("datetime").date.today().isoformat(),
        "data": os.path.basename(data_path)
    }
    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

        cases_md = sample_cases(
        config_path=config_path,
        data_path=data_path,
        num_cases=num_cases,
        seed=seed,
        return_md=True
    )
    with open(os.path.join(exp_dir, "cases.md"), "w", encoding="utf-8") as f:
        f.write(cases_md)

    print(f"[Done] metrics.json and cases.md saved in {exp_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    evaluate_model(args.config, args.data)



