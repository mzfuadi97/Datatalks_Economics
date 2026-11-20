"""
End-to-end pipeline orchestrator.

Runs the full algo-trading workflow:
  1. data_collector.py        - download & engineer features (raw)
  2. data_processor.py        - clean, target, save processed
  3. train_ml_models.py       - train models, save predictions
  4. simulate_trading.py      - backtest strategies
  5. portfolio_optimization.py- build optimised portfolios

Usage:
  python main.py                # run all steps
  python main.py --steps 3 4    # run selected steps
  python main.py --skip-data    # reuse existing raw data (file regime)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCRIPTS = BASE_DIR / "scripts"

STEP_SCRIPTS = {
    1: "data_collector.py",
    2: "data_processor.py",
    3: "train_ml_models.py",
    4: "simulate_trading.py",
    5: "portfolio_optimization.py",
}


def run_step(step: int) -> bool:
    script = SCRIPTS / STEP_SCRIPTS[step]
    print(f"\n{'='*60}\nSTEP {step}: {script.name}\n{'='*60}")
    start = time.time()
    result = subprocess.run([sys.executable, str(script)], cwd=str(BASE_DIR))
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"[FAIL] Step {step} exited with code {result.returncode}")
        return False
    print(f"[OK] Step {step} completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Stock trading ML pipeline")
    parser.add_argument("--steps", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5],
                        help="Steps to run (1-5)")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip step 1 if raw data already exists")
    args = parser.parse_args()

    steps = args.steps
    if args.skip_data and 1 in steps:
        steps = [s for s in steps if s != 1]
        # still need processing
        if 2 not in steps:
            steps = [2] + steps

    print(f"Running pipeline steps: {steps}")
    for step in steps:
        if step not in STEP_SCRIPTS:
            print(f"Unknown step {step}; valid steps: {list(STEP_SCRIPTS)}")
            return 1
        if not run_step(step):
            return 1

    print("\nPipeline finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
