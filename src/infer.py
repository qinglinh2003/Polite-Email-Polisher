import sys
import torch
from utils import load_model, get_tokenizer

def polite_rewrite(text: str, model_dir: str = "model/polite-rewriter", max_length: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    if len(sys.argv) < 2:
        print("Usage: python infer.py \"Your sentence here.\"")
        sys.exit(1)

    input_text = sys.argv[1]
    result = polite_rewrite(input_text)
    print("Polite Version:", result)
