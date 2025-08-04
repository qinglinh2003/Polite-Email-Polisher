import json
from tqdm import tqdm
import evaluate
from cli import load_config
from infer import polite_rewrite
from case_sampling import sample_cases 
from evaluation.politeness_scorer.scorer import PolitenessScorer
import os
import csv

def evaluate_model(config_path, num_cases=5, seed=42):
    config = load_config(config_path)
    data_path = config["data"]["test_path"]
    version = config["model"]["version"]
    exp_dir = os.path.join("experiments", version)
    os.makedirs(exp_dir, exist_ok=True)


    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    inputs = [item["input"] for item in data]
    references = [item["target"] for item in data]
    predictions = []

    for item in tqdm(data, desc="Evaluating"):
        pred = polite_rewrite(item["input"], config)
        predictions.append(pred)

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    bleu_result = bleu.compute(predictions=predictions, references=[[r] for r in references])
    rouge_result = rouge.compute(predictions=predictions, references=references, use_stemmer=True, use_aggregator=True)

    def _get_f1(score):
        return score.mid.fmeasure if hasattr(score, "mid") else float(score)

    scorer = PolitenessScorer(
    model_path="evaluation/politeness_scorer/checkpoints/politeness_scorer.ckpt",
    pretrained_model="xlm-roberta-base"
    )

    input_scores = scorer.score(inputs)
    prediction_scores = scorer.score(predictions)
    deltas = [pred - orig for pred, orig in zip(prediction_scores, input_scores)]
    avg_delta_politeness = sum(deltas) / len(deltas)
    
    num_total = len(deltas)
    num_improved = sum(1 for d in deltas if d > 0)
    improved_ratio = num_improved / num_total

    metrics = {
        "BLEU": bleu_result["bleu"],
        "ROUGE-1": _get_f1(rouge_result["rouge1"]),
        "ROUGE-L": _get_f1(rouge_result["rougeL"]),
        "Average Delta Politeness": avg_delta_politeness,
        "Improved Ratio": improved_ratio,
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

    csv_path = os.path.join(exp_dir, "politeness_comparison.csv")
    
    with open(csv_path, mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Original Sentence", "Original Politeness", "Polished Sentence", "Polished Politeness", "Delta"])

        for orig, orig_score, polished, polished_score in zip(inputs, input_scores, predictions, prediction_scores):
            delta = polished_score - orig_score
            writer.writerow([orig, f"{orig_score:.3f}", polished, f"{polished_score:.3f}", f"{delta:+.3f}"])


    print(f"[Done] metrics.json, cases.md, and politeness_comparison.csv saved in {exp_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    evaluate_model(args.config, args.data)



