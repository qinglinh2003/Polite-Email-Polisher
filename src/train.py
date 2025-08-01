from transformers import Seq2SeqTrainingArguments as TrainingArguments, Trainer
from data_loader import load_dataset_from_json
from utils import get_tokenizer, load_model, get_data_collator, tokenize_batch

def main(config):
    model_name = config["model"]["name_or_path"]
    data_path = config["data"]["train_path"]
    val_ratio = config["data"]["val_ratio"]
    max_length = config["data"]["max_length"]

    datasets = load_dataset_from_json(data_path, val_ratio)
    tokenizer = get_tokenizer(model_name)
    datasets = datasets.map(lambda x: tokenize_batch(tokenizer, x, max_length), batched=True, remove_columns=["input", "target"])

    model = load_model(model_name)
    data_collator = get_data_collator(tokenizer, model)

    training_args = TrainingArguments(
        output_dir=config["model"]["save_dir"],
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        logging_steps=config["training"]["logging_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        fp16=config["training"]["fp16"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()

    model.save_pretrained(config["model"]["save_dir"])
    tokenizer.save_pretrained(config["model"]["save_dir"])

if __name__ == "__main__":
    raise ValueError("Please run this script via cli.py using --mode train")


