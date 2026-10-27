"""Build an evidence-backed draft; never invent experiments or qualitative findings."""
import argparse
from pathlib import Path
import sys
from common import ROOT, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--evaluation", default="results/v2/evaluation.json")
    parser.add_argument("--generations", default="results/v2/generations.json")
    parser.add_argument("--predictions", default="results/v2/public.json")
    parser.add_argument("--source", default="reports/report_source.md")
    parser.add_argument("--output", default="reports/report")
    parser.add_argument("--v3", action="store_true")
    args = parser.parse_args()
    source = Path(args.source).read_text(encoding="utf-8")
    experiments = []
    files = ["prepared/baselines.json", "runs/training/history.json", args.evaluation]
    if args.v3:
        from run_ablations import summarize
        from v3_experiments import experiments as registry
        summarize()
        files = ["configs/v3/training.json", "configs/v3/selected.json", "configs/v3/models.lock.json",
                 "runs/v3/random/history.json", "runs/v3/hard/history.json",
                 "runs/v3/random/manifest.json", "runs/v3/hard/manifest.json", "results/v3/selection.json",
                 args.evaluation]
        table = (ROOT / "results/v3/ablations.csv").read_text().splitlines()
        table = [line.replace("NOT RUN", "UNEXECUTED OPTIONAL") if line.startswith(("B40,", "B120,")) else line
                 for line in table]
        experiments.append("Controlled comparison (train-validation exact metrics, not official public scores):\n\n" +
                           "\n".join(table))
        for key, entry in registry().items():
            path = ROOT / f"results/v3/ablations/{key}/result.json"
            if path.exists():
                record = read_json(path)
                import json
                experiments.append(key + " " + entry["name"] + "\n" + json.dumps({
                    "parent": entry["parent"], "config": record["config"],
                    "metrics": record["metrics"], "candidate_metrics": record["candidate_metrics"],
                    "timings": record["runtime"]["timings"], "cache_hit": record["runtime"]["cache_hit"]}, indent=2))
        experiments.append("Optional B40/B120 budget comparisons may remain unexecuted; "
                           "they are not candidates in the main selection search.")
    for filename in files:
        path = Path(filename)
        if path.exists():
            value = read_json(path)
            if filename == args.evaluation:
                value = {"official_metrics": value["metrics"], "judge": value["judge"]}
            elif args.v3 and filename.endswith("manifest.json"):
                value = {k: value[k] for k in ["selected_epoch", "selection", "beats_pretrained",
                         "parameter_counts", "runtime_seconds", "peak_allocated_gib", "pretrained_fingerprint"]}
            import json
            experiments.append(filename + "\n\n" + json.dumps(value, ensure_ascii=False, indent=2))
        else:
            experiments.append(filename + ": NOT RUN / result unavailable")
    generations = read_json(args.generations)
    if generations["prediction_sha256"] != sha256(args.predictions):
        raise ValueError("Generations came from different predictions")
    if len({s["qid"] for s in generations["samples"]}) < 5 or any(len(s["generations"]) != 3 for s in generations["samples"]):
        raise ValueError("Complete at least 5 samples x 3 prompts before building the report")
    blocks = []
    blocks += ["System prompt:", generations["system"], ""]
    for name, prompt in generations["prompts"].items():
        blocks += [f"Prompt definition ({name}):", prompt, ""]
    for sample in generations["samples"]:
        blocks += [f"### {sample['qid']} / {sample['pid']}", "", sample["conversation"],
                   "", f"Conversation truncated: {sample.get('conversation_truncated', 'not recorded')}; "
                   f"evidence truncated: {sample['evidence_truncated']}",
                   "Retrieved evidence:", sample["product_evidence"], ""]
        for name, generation in sample["generations"].items():
            blocks += [f"Prompt {name}; truncated={generation['truncated']}", generation["text"], ""]
    text = (source.replace("{{NAME}}", args.name).replace("{{STUDENT_ID}}", args.student_id)
            .replace("{{EXPERIMENTS}}", "\n\n".join(experiments))
            .replace("{{GENERATIONS}}", "\n".join(blocks)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".md").write_text(text, encoding="utf-8")
    from build_report import write_pdf
    write_pdf(text, str(output.with_suffix(".pdf")))
    write_json(output.with_suffix(".receipt.json"), {
        "pdf_sha256": sha256(output.with_suffix(".pdf")),
        "markdown_sha256": sha256(output.with_suffix(".md")),
        "source_sha256": sha256(args.source), "predictions_sha256": sha256(args.predictions),
        "generations_sha256": sha256(args.generations),
        "evaluation_sha256": sha256(args.evaluation) if Path(args.evaluation).exists() else None,
    })
    print(f"Wrote {output}.md and {output}.pdf")
    if "TODO_" in text:
        print(f"DRAFT: fill the discussion and qualitative analysis in {args.source}, then rebuild.")


if __name__ == "__main__":
    main()
