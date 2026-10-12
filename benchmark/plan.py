"""Plan on one registered environment and write the same result schema."""

from __future__ import annotations

import sys

from benchmark.command import execute, load_options
from benchmark.schema import write_result


def main(argv: list[str] | None = None) -> None:
    try:
        options = load_options(argv)
        record = execute(options, "plan")
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    write_result(options["output"], record)


if __name__ == "__main__":
    main(sys.argv[1:])
