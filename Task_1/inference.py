import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class MountainNER:
    def __init__(self, model_path="./weights/mountain_ner_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.nlp = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )

    def extract_mountains(self, text: str):
        ner_results = self.nlp(text)
        mountains = []

        for entity in ner_results:
            if "MOUNTAIN" in entity["entity_group"]:
                mountains.append({
                    "mountain": entity["word"].strip(),
                    "score": round(float(entity["score"]), 4),
                    "start": entity["start"],
                    "end": entity["end"]
                })
        return mountains


if __name__ == "__main__":
    ner_system = MountainNER()
    sample = "We started our trek at Mount Blanc and hope to finish at Matterhorn next week."
    results = ner_system.extract_mountains(sample)

    print("Text:", sample)
    if results:
        print("Detected Mountains:")
        for mountain in results:
            print(f"- {mountain['mountain']} (confidence: {mountain['score']:.2%})")
    else:
        print("No mountains detected.")