"""Build the seven-example CMCC training split without email-test leakage."""

from __future__ import annotations

import pickle
from pathlib import Path

PROCESSED_DIR = Path("benchmarks/cmcc/processed")
SOURCE_SPLITS = ("cmcc_train.pkl", "cmcc_val.pkl", "cmcc_test.pkl")
TEST_PATH = PROCESSED_DIR / "cmcc_email_2x3_test.pkl"
OUTPUT_PATH = PROCESSED_DIR / "cmcc_email_2x3_train_7_clean.pkl"
AUTHOR_KEYS = (4, 9)
TRAIN_EXAMPLES_PER_AUTHOR = 7


def load_pickle(path: Path) -> dict:
    with path.open("rb") as input_file:
        return pickle.load(input_file)


def main() -> None:
    test_data = load_pickle(TEST_PATH)
    source_data = [load_pickle(PROCESSED_DIR / filename) for filename in SOURCE_SPLITS]
    clean_train = {}

    for author_key in AUTHOR_KEYS:
        held_out = {(example["prompt"], example["output"]) for example in test_data[author_key]}
        candidates = []
        seen = set()
        for split in source_data:
            for example in split[author_key]:
                identity = (example["prompt"], example["output"])
                if identity in held_out or identity in seen:
                    continue
                candidates.append(example)
                seen.add(identity)

        if len(candidates) < TRAIN_EXAMPLES_PER_AUTHOR:
            raise ValueError(
                f"Author {author_key} has only {len(candidates)} non-test examples; "
                f"need {TRAIN_EXAMPLES_PER_AUTHOR}."
            )
        clean_train[author_key] = candidates[:TRAIN_EXAMPLES_PER_AUTHOR]

    with OUTPUT_PATH.open("wb") as output_file:
        pickle.dump(clean_train, output_file)

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
