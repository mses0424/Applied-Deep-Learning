# ADL 2026 HW3 — Tool-Calling LoRA for Qwen3-0.6B

My solution to Homework 3 of *Applied Deep Learning* (NTU, Fall 2026): one LoRA adapter on
`Qwen/Qwen3-0.6B-Base` that does both

- **Part 1 — tool calling:** decide whether to call a tool, pick the right function and fill
  its arguments, across multi-turn conversations with policies and tool results;
- **Part 2 — buyer negotiation:** act as a buyer agent that places correct orders at good prices.

The same weights serve both parts. `Qwen/Qwen3-0.6B` is used only for its chat tokenizer and
template.

## Results (complete PUBLIC test sets, official scorer)

| Metric | Value |
|---|---:|
| Part 1 weighted step accuracy (first-step rule, 13,021 steps) | **68.40%** |
| Part 1 ordinary step accuracy | 72.58% |
| Part 2 order accuracy (2,607 negotiations) | **85.49%** |
| Part 2 fill rate | **87.88%** |
| Part 2 relative price (lower is better) | 1.050 |

Untuned base model on the same first 100 PUBLIC steps: 0.52% weighted accuracy (adapter: 68.04%).
Score files are in [`code/evidence/`](code/evidence/); the full write-up is in
[`report/report.pdf`](report/report.pdf).

## Method in short

- **Data (v12 mix, 49,349 training / 2,218 held-out conversations):** public *training* splits
  of BFCL (including trajectories replayed in the official simulator), APIGen-MT, Synth-APIGen,
  When2Call, ToolACE and Toucan, converted to Qwen tool-call messages; plus original synthetic
  data: policy-driven conversations, multi-tool dependency chains, no-call cases and buyer
  negotiations. Candidate rows that overlap the homework PUBLIC text are discarded
  (`public_guard_v10.py`); PUBLIC examples and labels are never used for training.
- **Training:** LoRA rank 128 / alpha 192 / dropout 0.05 on all linear layers, max length 3,072,
  lr 1.2e-4 with cosine decay and 4% warmup, 2 epochs, effective batch 16, FP16 on 2 x RTX 2080 Ti.
- **Selection:** retained checkpoints are scored on a fixed 600-step PUBLIC subset with the
  official scorer; step 10,800 was selected and then evaluated on the complete PUBLIC sets.

## Repository layout

```
code/                 all data preparation, training, evaluation and report scripts
  README.md           what every file does
  evidence/           score JSONs and learning-curve data behind the report
report/report.pdf     final report
docs/SUBMISSION_README.md   the README handed in with the course submission (full commands)
```

Not included:

- **The adapter weights** (`adapter_checkpoint/`, 309 MB, above GitHub's 100 MB file limit).
  Attach them to a GitHub Release or upload them to the Hugging Face Hub if you want to share them.
- **Course materials** (`eval.py`, `eval/`, `data/`, the starter pack): they belong to the course.
- Generated training data, checkpoints and environments (tens of GB).

## Reproducing

The scripts expect to run inside the original HW3 folder, next to the course-provided
`eval.py`, `eval/`, `requirements.txt` and `data/public_test*.jsonl.gz`. In short:

```bash
bash code/create_ta_venv.sh                          # Python 3.11 env (torch 2.11, transformers 5.13, peft 0.19, vllm 0.25)
.venv-ta/bin/python code/download_v12_sources.py     # pinned public training sources
bash code/prepare_v12_data.sh                        # -> training_data_fresh_v12/
RUN_TAG=reproduce_v12 GPU_IDS=0,1 bash code/launch_fresh_v12.sh   # train, select, evaluate
```

Step-by-step instructions (environment check, data audit, evaluation of the adapter with the
staff scorer, report rebuild) are in [`docs/SUBMISSION_README.md`](docs/SUBMISSION_README.md).

## Author

連家堯 (r15922139), National Taiwan University.
