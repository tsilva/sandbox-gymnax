from __future__ import annotations

import sys

from sandbox_gymnax.train import *  # noqa: F403
from sandbox_gymnax.train import main as train_main


def main() -> None:
    if "--env-id" not in sys.argv:
        sys.argv.extend(["--env-id", "CartPole-v1"])
    train_main()


if __name__ == "__main__":
    main()
