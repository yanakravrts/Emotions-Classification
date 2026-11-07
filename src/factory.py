from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

def create_model(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    return tokenizer, model