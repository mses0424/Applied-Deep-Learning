"""Run the untouched official evaluator, with optional small compatibility smoke subset."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from common import ROOT, read_json, read_jsonl, write_json, write_jsonl, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="results/v2/public.json")
    parser.add_argument("--dialogues", default="data/public/dialogues.jsonl")
    parser.add_argument("--products", default="data/public/products.jsonl")
    parser.add_argument("--output", default="results/v2/evaluation.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--smoke", type=int, default=0, help="Judge first N queries; never used for pass/fail")
    args = parser.parse_args()
    os.chdir(ROOT)
    if Path(args.output).exists():
        raise SystemExit("Evaluation output exists; use a new path.")
    predictions, dialogues = args.predictions, args.dialogues
    if args.smoke:
        if args.smoke < 1:
            raise ValueError("Smoke count must be positive")
        rows = read_jsonl(dialogues)[:args.smoke]
        selected = {r["dialogue_id"] for r in rows}
        payload = read_json(predictions)
        payload["results"] = [r for r in payload["results"] if r["qid"] in selected]
        folder = Path(args.output).parent / (Path(args.output).stem + "_inputs")
        predictions, dialogues = str(folder / "predictions.json"), str(folder / "dialogues.jsonl")
        write_json(predictions, payload)
        write_jsonl(dialogues, rows)
    subprocess.run([sys.executable, "evaluate.py", "--retrieval-results", predictions,
                    "--dialogues", dialogues, "--products", args.products,
                    "--model", "models/Qwen2.5-1.5B-Instruct", "--device", args.device,
                    "--batch-size", str(args.batch_size), "--output", args.output], check=True)
    write_json(Path(args.output).with_suffix(".receipt.json"), {
        "evaluation_sha256": sha256(args.output), "predictions_sha256": sha256(args.predictions),
        "dialogues_sha256": sha256(args.dialogues), "products_sha256": sha256(args.products),
        "evaluate_py_sha256": sha256("evaluate.py"), "smoke": args.smoke,
        "device": args.device, "precision": "official unchanged BF16",
    })


if __name__ == "__main__":
    main()
