"""Audit one registered environment.

``--split test`` is accepted only when that environment declares a test split.
The arm and the box pusher do not. Their validation tails are not relabeled.
"""

from __future__ import annotations

import sys

from benchmark.command import execute, load_options
from benchmark.schema import write_result


def main(argv: list[str] | None = None) -> None:
    try:
        options = load_options(argv)
        record = execute(options, "audit")
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    write_result(options["output"], record)


if __name__ == "__main__":
    main(sys.argv[1:])
