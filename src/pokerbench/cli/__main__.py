"""Allow `python -m pokerbench.cli`."""

from pokerbench.cli.main import (
    main,
)


if __name__ == "__main__":
    raise SystemExit(
        main()
    )