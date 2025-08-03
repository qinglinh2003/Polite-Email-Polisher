import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer

class TydiPDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        df = pd.read_csv(file_path)
        self.sentences = df['sentence'].tolist()
        self.scores = df['score'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        score = self.scores[idx]
        encoded = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0), torch.tensor(score, dtype=torch.float)

class TydiPDataModule(pl.LightningDataModule):
    def __init__(self, train_file, test_file, tokenizer_name, batch_size, max_length):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage=None):
        self.train_dataset = TydiPDataset(self.train_file, self.tokenizer, self.max_length)
        self.test_dataset = TydiPDataset(self.test_file, self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
