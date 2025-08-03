import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

class PolitenessRegressor(pl.LightningModule):
    def __init__(self, pretrained_model, learning_rate, num_warmup_steps):
        super().__init__()
        self.save_hyperparameters()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        score = self.regressor(cls_output).squeeze(-1)
        return score

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        preds = self(input_ids, attention_mask)
        loss = nn.MSELoss()(preds, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        preds = self(input_ids, attention_mask)
        loss = nn.MSELoss()(preds, labels)
        self.log('val_loss', loss)
        return {"preds": preds, "labels": labels}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
