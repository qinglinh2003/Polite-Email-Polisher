import yaml
import pytorch_lightning as pl
from models.politeness_regressor import PolitenessRegressor
from datamodule.tydip_datamodule import TydiPDataModule
import os


config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'scorer_config.yaml'))
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

pl.seed_everything(config['train']['seed'])

model = PolitenessRegressor(
    pretrained_model=config['model']['pretrained_model'],
    learning_rate=config['model']['learning_rate'],
    num_warmup_steps=config['train']['num_warmup_steps']
)

data_module = TydiPDataModule(
    train_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'en_train.csv')),
    test_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'en_test.csv')),
    tokenizer_name=config['model']['pretrained_model'],
    batch_size=config['model']['batch_size'],
    max_length=config['model']['max_length']
)

trainer = pl.Trainer(max_epochs=config['train']['epochs'], precision=16)
trainer.fit(model, data_module)
trainer.save_checkpoint('checkpoints/politeness_scorer.ckpt')
