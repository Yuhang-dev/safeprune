from __future__ import annotations

import argparse

from safeprune.config import load_config
from safeprune.training import recover_with_lora


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    recover_with_lora(load_config(args.config))


if __name__ == "__main__":
    main()

