"""Evaluate the frozen v3 selection on all public queries; no training or tuning."""
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from common import ROOT, read_json, write_json, sha256, model_fingerprint


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["v3", "v4"], default="v4")
    version = parser.parse_args().version
    os.chdir(ROOT)
    predictions = ROOT / f"results/{version}/public.json"
    evaluation = ROOT / f"results/{version}/evaluation.json"
    config = ROOT / f"configs/{version}/selected.json"
    subprocess.run([sys.executable, "code/script/check_sources.py"], check=True)
    if evaluation.exists():
        raise SystemExit("Existing official evaluation preserved; choose a separate run explicitly.")
    state_path = ROOT / f"results/{version}/public_job.json"
    state = {"status": "running", "stage": "retrieval", "started_at": datetime.now().astimezone().isoformat(),
             "config_sha256": sha256(config), "judge_batch_size": 1,
             "selection": read_json(ROOT / f"results/{version}/selection.json")["selected"]}
    write_json(state_path, state)
    try:
        # Ignore interactive experiment overrides; measure exactly the selected method.
        env = {k: v for k, v in os.environ.items() if not k.startswith("HW2_")}
        env.update(HW2_DEVICE="cuda:0", TOKENIZERS_PARALLELISM="false")
        if not predictions.exists():
            (ROOT / "cache").mkdir(exist_ok=True)
            cache = tempfile.mkdtemp(prefix=f"public_{version}_cold_", dir=ROOT / "cache")
            env["HW2_CACHE_DIR"] = cache
            print("[1/3] Full public retrieval, NEW cold cache:", cache, flush=True)
            subprocess.run([sys.executable, "code/script/run_retrieval.py", "--config", str(config),
                            "--output", str(predictions)], env=env, check=True)
        else:
            receipt = read_json(predictions.with_suffix(".run.json"))
            checks = {"prediction_sha256": sha256(predictions), "config_sha256": sha256(config),
                      "retrieval_source_sha256": sha256(ROOT / "retrieval_utils.py"),
                      "dialogues_sha256": sha256(ROOT / "data/public/dialogues.jsonl"),
                      "products_sha256": sha256(ROOT / "data/public/products.jsonl")}
            if any(receipt[k] != v for k, v in checks.items()):
                raise ValueError("Existing retrieval provenance differs; refusing to mix runs")
            cfg = read_json(config)
            for name, digest in receipt["model_fingerprints"].items():
                if model_fingerprint(ROOT / cfg[name + "_model"]) != digest:
                    raise ValueError("Model changed since existing retrieval")
            print("[1/3] Reusing verified complete public retrieval", flush=True)
        state["stage"] = "official_judge"
        write_json(state_path, state)
        print("[2/3] Original BF16 Qwen judge, ALL 1000 queries, batch=1 (VRAM safety)", flush=True)
        subprocess.run([sys.executable, "code/script/run_evaluation.py", "--predictions", str(predictions),
                        "--output", str(evaluation), "--batch-size", "1"], env=env, check=True)
        print("[3/3] Checking official thresholds and measured retrieval time", flush=True)
        result = subprocess.run([sys.executable, "code/script/check_targets.py", "--predictions", str(predictions),
                                 "--evaluation", str(evaluation), "--target", "boss"],
                                env=env, capture_output=True, text=True)
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", file=sys.stderr, flush=True)
        if result.returncode not in (0, 1) or "INVALID:" in result.stdout + result.stderr:
            raise RuntimeError("Threshold checker rejected evaluation provenance")
        state.update(status="completed", stage="finished", metrics=read_json(evaluation)["metrics"],
                     boss_and_local_time_pass=result.returncode == 0, threshold_check=result.stdout,
                     finished_at=datetime.now().astimezone().isoformat())
        write_json(state_path, state)
        print("PUBLIC EVALUATION COMPLETE (completion alone does not imply boss PASS)", flush=True)
    except BaseException as error:
        state.update(status="failed", error=repr(error), finished_at=datetime.now().astimezone().isoformat())
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
