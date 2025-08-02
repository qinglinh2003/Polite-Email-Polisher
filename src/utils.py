from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

def get_tokenizer(tokenizer_name: str):
    return AutoTokenizer.from_pretrained(tokenizer_name)

def load_model(model_name_or_path: str):
    """
    Load a Seq2Seq model from Hugging Face model hub or local path.
    """
    return AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

def get_data_collator(tokenizer, model):
    """
    Return a DataCollator that pads inputs to the correct length.
    """
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

def tokenize_batch(tokenizer, batch, max_length=64):
    """
    Tokenize a batch of input-target pairs.
    Args:
        batch: Dict with keys 'input', 'target'
    Returns:
        Dict with tokenized input_ids, attention_mask, labels
    """
    inputs = tokenizer(batch["input"], padding="max_length", truncation=True, max_length=max_length)
    targets = tokenizer(batch["target"], padding="max_length", truncation=True, max_length=max_length)

    labels = []
    for target_ids in targets["input_ids"]:
        labels.append([
            token_id if token_id != tokenizer.pad_token_id else -100
            for token_id in target_ids
        ])

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels  
    }

  
