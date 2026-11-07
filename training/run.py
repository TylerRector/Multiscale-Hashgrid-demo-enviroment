"""Generate supervision, train, validate, and export a hashgrid model."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["fix_p5", "micro"], default="fix_p5")
    parser.add_argument("--run", default="demo_training")
    parser.add_argument("--snapshot-every", type=int, default=0)
    args = parser.parse_args()
    if not args.run or Path(args.run).name != args.run or args.run in (".", ".."):
        parser.error("--run must be a directory name")
    stages = [
        ("stage1_trace/trace.py", True),
        ("stage2_gather/gather.py", True),
        ("stage2_gather/check_gather.py", True),
        ("stage3_train/train.py", True),
        ("stage3_train/validate.py", True),
        ("stage4_export/export.py", False),
        ("stage4_export/check_parity.py", False),
    ]
    for script, needs_profile in stages:
        command = [sys.executable, "-u", str(ROOT / script), "--run", args.run]
        if needs_profile:
            command += ["--profile", args.profile]
        if script == "stage3_train/train.py":
            command += ["--snapshot-every", str(args.snapshot_every)]
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"Export: {ROOT / 'runs' / args.run / 'export'}")
    print(f"Play: ./build/hashgrid_demo --model training/runs/{args.run}/export")


if __name__ == "__main__":
    main()
