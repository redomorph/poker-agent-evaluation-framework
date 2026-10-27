#!/usr/bin/env python
"""
Run the validation test file repeatedly, stopping on first failure.

Usage: python scripts/run_validation_loop.py [--runs N]

Default N = 50.
"""

import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TEST_FILE = PROJECT_ROOT / "tests" / "validation" / "test_phase0_gates.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50, help="Number of runs to attempt")
    args = parser.parse_args()

    if not TEST_FILE.exists():
        print(f"Test file not found: {TEST_FILE}")
        return 1

    print(f"Running validation tests up to {args.runs} times, stopping on first failure.")
    print(f"Test file: {TEST_FILE}\n")

    for i in range(1, args.runs + 1):
        print(f"Run {i}/{args.runs}...", end="", flush=True)
        cmd = [
            sys.executable, "-m", "pytest",
            str(TEST_FILE),
            "-v", "--tb=short",  # short traceback for readability
            "--no-header",        # reduce noise
        ]
        # We run in the project root using the same Python interpreter (poetry env)
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

        if result.returncode == 0:
            print(" OK")
            continue

        # Failure
        print(" FAILED")
        print("\n" + "=" * 80)
        print(f"Test failed on run {i} with exit code {result.returncode}")
        print("-- stdout --")
        print(result.stdout)
        print("-- stderr --")
        print(result.stderr)
        print("=" * 80)
        return 1

    print(f"\nAll {args.runs} runs passed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())