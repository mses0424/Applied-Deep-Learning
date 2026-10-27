"""Build a measured v4 report draft; missing public/Q2 evidence stays explicitly unfinished."""
import argparse
import json
from common import ROOT, read_json, write_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="TODO_NAME")
    parser.add_argument("--student-id", default="TODO_STUDENT_ID")
    args = parser.parse_args()
    from v4_pipeline import summarize
    summarize()
    source = ROOT / "reports/v4/report_source.md"
    blocks = [(ROOT / "results/v4/comparisons.csv").read_text()]
    for name in ["configs/v4/training.json", "results/v4/candidate_selection.json", "results/v4/selection.json",
                 "runs/v4/listwise/history.json", "runs/v4/pairwise/history.json", "configs/v4/selected.json"]:
        path = ROOT / name
        blocks.append(name + "\n" + (json.dumps(read_json(path), indent=2) if path.exists() else "NOT RUN"))
    for mode in ("listwise", "pairwise"):
        path = ROOT / f"runs/v4/{mode}/manifest.json"
        if path.exists():
            m = read_json(path)
            blocks.append(json.dumps({"mode": mode, **{k: m[k] for k in
                ("selected_epoch", "trained_epochs", "parameter_counts", "peak_allocated_gib", "runtime_seconds")}}, indent=2))
    evaluation = ROOT / "results/v4/evaluation.json"
    predictions = ROOT / "results/v4/public.json"
    generations = ROOT / "results/v4/generations.json"
    if evaluation.exists():
        e = read_json(evaluation)
        blocks.append("Official PUBLIC evaluation:\n" + json.dumps({k: e[k] for k in ("metrics", "judge")}, indent=2))
    else:
        blocks.append("Official PUBLIC evaluation: NOT RUN; train exact metrics cannot establish boss passing.")
    generation_blocks = ["Q2: NOT RUN"]
    if generations.exists():
        g = read_json(generations)
        if g["prediction_sha256"] != sha256(predictions): raise ValueError("Generation/prediction mismatch")
        if len({s["qid"] for s in g["samples"]}) < 5 or any(len(s["generations"]) != 3 for s in g["samples"]):
            raise ValueError("Finish five distinct samples and three prompts each")
        generation_blocks = ["System prompt:", g["system"], json.dumps(g["prompts"], indent=2)]
        for sample in g["samples"]:
            generation_blocks += [f"Sample {sample['qid']}, product {sample['pid']}", sample["conversation"],
                                  "Supplied product evidence:", sample["product_evidence"],
                                  f"Evidence truncated: {sample['evidence_truncated']}; conversation truncated: {sample.get('conversation_truncated')}"]
            for key, value in sample["generations"].items():
                generation_blocks += [f"Prompt {key}, output truncated: {value['truncated']}", value["text"]]
    text = source.read_text().replace("{{NAME}}", args.name).replace("{{STUDENT_ID}}", args.student_id)
    text = text.replace("{{EXPERIMENTS}}", "\n\n".join(blocks)).replace("{{GENERATIONS}}", "\n\n".join(generation_blocks))
    target = ROOT / "reports/v4/report"
    target.with_suffix(".md").write_text(text, encoding="utf-8")
    from build_report import write_pdf
    write_pdf(text, str(target.with_suffix(".pdf")))
    write_json(target.with_suffix(".receipt.json"), {
        "pdf_sha256": sha256(target.with_suffix(".pdf")), "markdown_sha256": sha256(target.with_suffix(".md")),
        "predictions_sha256": sha256(predictions) if predictions.exists() else None,
        "evaluation_sha256": sha256(evaluation) if evaluation.exists() else None,
        "generations_sha256": sha256(generations) if generations.exists() else None,
        "selection_sha256": sha256(ROOT / "results/v4/selection.json") if (ROOT / "results/v4/selection.json").exists() else None,
        "source_sha256": sha256(source)})
    print("Built reports/v4/report.md and .pdf. TODO/NOT RUN entries mean DRAFT, not ready to submit.")


if __name__ == "__main__":
    main()
