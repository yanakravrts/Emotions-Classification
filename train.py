import argparse
import yaml

from src.factory import create_model
from src.trainer import train_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["bert", "roberta", "distilbert", "deberta", "albert"],
    )

    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    model_name = config["models"][args.model]

    tokenizer, model = create_model(
        model_name=model_name,
        num_labels=config["dataset"]["num_labels"],
    )

    train_model(
        model=model,
        tokenizer=tokenizer,
        config=config,
        model_key=args.model,
    )


if __name__ == "__main__":
    main()