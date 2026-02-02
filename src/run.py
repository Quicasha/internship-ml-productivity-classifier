import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


TASK_TO_SCRIPT = {
    ("train", "logreg"): "train_logistic.py",
    ("train", "rf"): "train_random_forest.py",
    ("featimp", None): "feature_importance.py",
    ("ablation", None): "ablation_plot.py",
    ("realtime", None): "realtime_simulation.py",
}


def run_script(script_name: str) -> int:
    script_path = SRC_DIR / script_name
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return 1

    # Use the same Python interpreter that is running this file
    cmd = [sys.executable, str(script_path)]
    print(f"[RUN] {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Project runner: train/eval/analysis tasks."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["train", "featimp", "ablation", "realtime"],
        help="What to run",
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "rf"],
        default=None,
        help="Model for training task (logreg or rf)",
    )

    args = parser.parse_args()

    key = (args.task, args.model if args.task == "train" else None)
    script = TASK_TO_SCRIPT.get(key)

    if script is None:
        print("[ERROR] Invalid combination.")
        if args.task == "train":
            print("Use: --task train --model logreg|rf")
        else:
            print("Use: --task featimp OR --task ablation OR --task realtime")
        sys.exit(1)

    code = run_script(script)
    sys.exit(code)


if __name__ == "__main__":
    main()