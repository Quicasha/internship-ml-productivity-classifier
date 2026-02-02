"""
Main entrypoint for running experiments.

"""

import argparse
from sklearn.model_selection import train_test_split

from preprocess import prepare_dataset
from train_random_forest import build_model as build_rf
from train_logistic import build_model as build_logreg
from train_dummy import build_model as build_dummy
from metrics import pretty_print


# Registry of available models
MODEL_REGISTRY = {
    "rf": build_rf,
    "logreg": build_logreg,
    "dummy": build_dummy,
}


def train_holdout(model_key: str, test_size: float, seed: int):
    """
    Train and evaluate a single model using a holdout split.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}")

    X, y = prepare_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    model = MODEL_REGISTRY[model_key]()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\n=== {model_key.upper()} (holdout) ===")
    pretty_print(y_test, y_pred)


def compare_models(test_size: float, seed: int):
    """
    Train and compare all models on the same holdout split.
    """
    X, y = prepare_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    for name, builder in MODEL_REGISTRY.items():
        model = builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n=== {name.upper()} ===")
        pretty_print(y_test, y_pred)


def main():
    parser = argparse.ArgumentParser(description="ML experiment runner")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    train_p = subparsers.add_parser("train")
    train_p.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys())
    train_p.add_argument("--test-size", type=float, default=0.3)
    train_p.add_argument("--seed", type=int, default=42)

    # ---- compare ----
    compare_p = subparsers.add_parser("compare")
    compare_p.add_argument("--test-size", type=float, default=0.3)
    compare_p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "train":
        train_holdout(
            model_key=args.model,
            test_size=args.test_size,
            seed=args.seed,
        )

    elif args.command == "compare":
        compare_models(
            test_size=args.test_size,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()