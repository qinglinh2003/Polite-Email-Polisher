import torch
from transformers import AutoTokenizer
from .models.politeness_regressor import PolitenessRegressor

class PolitenessScorer:
    def __init__(self, model_path, pretrained_model="xlm-roberta-base", max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = PolitenessRegressor.load_from_checkpoint(model_path, pretrained_model=pretrained_model, learning_rate=1e-5, num_warmup_steps=0).cuda().eval()
        self.max_length = max_length

    def score(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        inputs = self.tokenizer(sentences, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to('cuda')
        with torch.no_grad():
            scores = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return scores.cpu().numpy().tolist()
