from __future__ import annotations

import argparse

from safeprune.data import load_preference_jsonl, summarize_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    records = load_preference_jsonl(args.path)
    print(summarize_records(records))


if __name__ == "__main__":
    main()

