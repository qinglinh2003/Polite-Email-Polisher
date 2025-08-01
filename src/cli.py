import argparse
import yaml
from train import main as train_main
from infer import polite_rewrite

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_train(config):
    train_main(config)

def run_infer(config, text):
    result = polite_rewrite(text, config)
    print("Polite:", result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--mode", choices=["train", "infer", "evaluate"], required=True)
    parser.add_argument("--text", type=str, help="Text input for inference mode")
    parser.add_argument("--num_cases", type=int, default=5, help="Number of cases for qualitative sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "train":
        run_train(config)
    elif args.mode == "infer":
        if not args.text:
            raise ValueError("Please provide --text for inference mode.")
        run_infer(config, args.text)
    elif args.mode == "evaluate":
        from evaluate_model import evaluate_model
        evaluate_model(
            config_path=args.config,
            num_cases=args.num_cases,
            seed=args.seed
        )

if __name__ == "__main__":
    main()
