import json
from datasets import Dataset, DatasetDict

def load_dataset_from_json(data_path: str, val_ratio: float = 0.1) -> DatasetDict:
    """
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = Dataset.from_list(data)
    split = dataset.train_test_split(test_size=val_ratio, seed=42)

    return DatasetDict({
        "train": split["train"],
        "validation": split["test"]
    })

if __name__ == "__main__":
    ds = load_dataset_from_json("data/polite_pairs.json", val_ratio=0.1)
    print(ds)