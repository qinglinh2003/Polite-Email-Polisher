import torch
from transformers import AutoTokenizer
from evaluation.politeness_scorer.models.politeness_regressor import PolitenessRegressor
import argparse
import pandas as pd

def load_model(checkpoint_path, pretrained_model):
    model = PolitenessRegressor.load_from_checkpoint(checkpoint_path, pretrained_model=pretrained_model, learning_rate=1e-5, num_warmup_steps=0)
    model.eval()
    return model

def predict(model, tokenizer, sentences, max_length=128):
    inputs = tokenizer(sentences, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    inputs = {k: v.cuda() for k, v in inputs.items()}
    model = model.cuda()
    with torch.no_grad():
        scores = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    return scores.cpu().numpy().tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/politeness_scorer.ckpt")
    parser.add_argument("--pretrained_model", type=str, default="xlm-roberta-base")
    parser.add_argument("--input_file", type=str, help="CSV file with 'sentence' column", default=None)
    parser.add_argument("--sentence", type=str, help="Single sentence for scoring", default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = load_model(args.model_path, args.pretrained_model)

    if args.sentence:
        sentences = [args.sentence]
    elif args.input_file:
        df = pd.read_csv(args.input_file)
        sentences = df['sentence'].tolist()
    else:
        raise ValueError("Provide either --sentence or --input_file")

    scores = predict(model, tokenizer, sentences)
    for s, score in zip(sentences, scores):
        print(f"Sentence: {s}")
        print(f"Politeness Score: {score:.3f}")
