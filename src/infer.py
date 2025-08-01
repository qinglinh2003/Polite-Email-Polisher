import sys
import torch
from utils import load_model, get_tokenizer

def polite_rewrite(text: str, config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = config["model"]["save_dir"]
    max_length = config["data"]["max_length"]
    num_beams = config["inference"]["num_beams"]

    tokenizer = get_tokenizer(model_dir)
    model = load_model(model_dir).to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=5,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    raise ValueError("Please run this script via cli.py using --mode infer --text '...'")

