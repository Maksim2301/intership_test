import os
import json
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import evaluate

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

LABEL_LIST = ["O", "B-MOUNTAIN", "I-MOUNTAIN"]
label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label = {i: label for i, label in enumerate(LABEL_LIST)}

MODEL_NAME = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_ner_tags = []
    for item in data:
        sample_tags = []
        for tag in item["ner_tags"]:
            if tag in ["B-LOCATION", "B-LOC", "B-GEO"]:
                tag = "B-MOUNTAIN"
            elif tag in ["I-LOCATION", "I-LOC", "I-GEO"]:
                tag = "I-MOUNTAIN"
            if tag not in label2id:
                tag = "O"

            sample_tags.append(label2id[tag])
        cleaned_ner_tags.append(sample_tags)

    return Dataset.from_dict({
        "tokens": [item["tokens"] for item in data],
        "ner_tags": cleaned_ner_tags
    })


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )
    labels = []

    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(
                    label2id["I-MOUNTAIN"] if label[word_idx] == label2id["B-MOUNTAIN"] else label[word_idx]
                )
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


seqeval = evaluate.load("seqeval")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [LABEL_LIST[p] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    true_labels = [
        [LABEL_LIST[l] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    return seqeval.compute(
        predictions=true_predictions,
        references=true_labels
    )


def main():
    print("Loading datasets...")
    train_dataset = load_data("data/train.json").map(tokenize_and_align_labels, batched=True)
    val_dataset = load_data("data/val.json").map(tokenize_and_align_labels, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        num_train_epochs=5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics
    )

    print("Starting training...")
    trainer.train()

    output_dir = "./weights/mountain_ner_model"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nTraining complete. Best fine-tuned model saved to {output_dir}")


if __name__ == "__main__":
    main()