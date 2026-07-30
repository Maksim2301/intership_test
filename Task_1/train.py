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

# Define NER target schema and index mappings
LABEL_LIST = ["O", "B-MOUNTAIN", "I-MOUNTAIN"]
label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label = {i: label for i, label in enumerate(LABEL_LIST)}

# Base pre-trained model checkpoint
MODEL_NAME = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_data(filepath):
    """Loads JSON dataset, maps generic location tags to MOUNTAIN, and returns a Hugging Face Dataset"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_ner_tags = []
    for item in data:
        sample_tags = []
        for tag in item["ner_tags"]:
            # Remap legacy/generic location labels to target MOUNTAIN tags
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
    """Tokenizes word-level inputs and aligns labels with sub-word tokens."""
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
            # Mask special tokens ([CLS], [SEP], [PAD]) with -100 to ignore in loss computation
            if word_idx is None:
                label_ids.append(-100)
            # Assign label to the first sub-token of a word
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For subsequent sub-tokens, convert B-MOUNTAIN to I-MOUNTAIN or preserve tag
            else:
                label_ids.append(
                    label2id["I-MOUNTAIN"] if label[word_idx] == label2id["B-MOUNTAIN"] else label[word_idx]
                )
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Load seqeval metric for entity-level evaluation
seqeval = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    """Calculates seqeval metrics by filtering out masked tokens"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Convert predictions back to string labels, ignoring masked positions
    true_predictions = [
        [LABEL_LIST[p] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    # Convert ground truth back to string labels, ignoring masked positions
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
    # Load and preprocess training and validation sets
    train_dataset = load_data("data/train.json").map(tokenize_and_align_labels, batched=True)
    val_dataset = load_data("data/val.json").map(tokenize_and_align_labels, batched=True)

    # Initialize model with custom label configuration
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id
    )

    # Configure hyperparameter search and training settings
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        num_train_epochs=5,
    )

    # Instantiate Trainer engine with dynamic padding data collator
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

    # Save fine-tuned model weights and tokenizer for inference
    output_dir = "./weights/mountain_ner_model"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nTraining complete. Best fine-tuned model saved to {output_dir}")

if __name__ == "__main__":
    main()