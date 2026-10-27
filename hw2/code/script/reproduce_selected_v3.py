"""Reproduce original v3 training if v4's final selection falls back to its checkpoint."""
import os
import shutil
import subprocess
from common import ROOT, read_json


def main():
    chosen = read_json(ROOT / "configs/v4/selected.json")
    if chosen["reranker_model"] != "models/v3-reranker-hard":
        return
    target = ROOT / "reproduction_v3"
    original = ROOT / "archive/v3_20261007"
    if not target.exists():
        target.mkdir()
        for name in ("configs", "scripts"):
            shutil.copytree(original / name, target / name, ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("retrieval_utils.py", "retrieval.py", "eval_utils.py", "evaluate.py", "run_v3.sh"):
            shutil.copy2(original / name, target / name)
        (target / "models").symlink_to(ROOT / "models", target_is_directory=True)
        (target / "data").symlink_to(ROOT / "data", target_is_directory=True)
    elif not (target / "run_v3.sh").exists():
        raise ValueError("Unrecognized reproduction directory; refusing to overwrite")
    env = dict(os.environ, HF_HOME=str(ROOT / ".cache/huggingface"))
    subprocess.run(["bash", "run_v3.sh"], cwd=target, env=env, check=True)


if __name__ == "__main__":
    main()
