"""
CPU inference for the fine-tuned DistilBERT ticket classifier.

Loads model from section3_classifier/model/, predicts on raw text,
returns one of {billing, technical_issue, feature_request, complaint, other}.
"""
import json
from pathlib import Path
from typing import List, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path(__file__).parent / "model"


class TicketClassifier:
    def __init__(self, model_dir: Path = MODEL_DIR):
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model not found at {model_dir}. "
                "Train via the Colab notebook and place the unzipped model_export here."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        # Force CPU — task spec requires CPU inference
        self.device = torch.device("cpu")
        self.model.to(self.device)
        with open(model_dir / "label_map.json", "r") as f:
            label_map = json.load(f)
        self.id2label = {int(k): v for k, v in label_map["id2label"].items()}

    @torch.no_grad()
    def predict(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Predict class for a single string or list of strings."""
        single = isinstance(text, str)
        texts = [text] if single else text
        inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        labels = [self.id2label[pid] for pid in pred_ids]
        return labels[0] if single else labels


if __name__ == "__main__":
    clf = TicketClassifier()
    samples = [
        "I was charged twice for my subscription this month",
        "The export button doesn't work, it just hangs",
        "Can you please add a dark mode to the dashboard?",
        "Your customer service is the worst I've ever experienced",
        "What are your office hours in India?",
    ]
    for s in samples:
        print(f"  [{clf.predict(s):16s}] {s}")


