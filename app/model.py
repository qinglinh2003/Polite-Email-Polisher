import yaml
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model():
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    version = config["model"]["version"]
    model_path = os.path.join("model", version)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return {"tokenizer": tokenizer, "model": model}

def polite_rewrite(text, model_bundle):
    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result
