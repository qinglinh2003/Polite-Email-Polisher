from transformers import Seq2SeqTrainingArguments as TrainingArguments, Trainer
from data_loader import load_dataset_from_json
from utils import get_tokenizer, load_model, get_data_collator, tokenize_batch

def main():
    model_name = "google/flan-t5-small"
    data_path = "data/polite_pairs.json"

    datasets = load_dataset_from_json(data_path, val_ratio=0.1)
    tokenizer = get_tokenizer(model_name)
    datasets = datasets.map(lambda x: tokenize_batch(tokenizer, x), batched=True, remove_columns=["input", "target"])
    model = load_model(model_name)
    data_collator = get_data_collator(tokenizer, model)

    training_args = TrainingArguments(
        output_dir="model/polite-rewriter",
        num_train_epochs=8,
        per_device_train_batch_size=8,
        learning_rate=3e-5,
        logging_steps=50,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=True
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

    model.save_pretrained("model/polite-rewriter")
    tokenizer.save_pretrained("model/polite-rewriter")

if __name__ == "__main__":
    main()


