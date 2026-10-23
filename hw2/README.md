# ADL 2026 HW2 — r15922139

This README explains how to set up the project, download the models, reproduce
the submitted reranker, run retrieval, and locate the main implementation files.
Run every command from the directory containing this README.

## TA quick path

In the prepared course environment, the required student entrypoints are:

```bash
bash download.sh
bash run.sh
```

`download.sh` only downloads the pretrained checkpoints and the selected trained
reranker. It does not install Python packages. `run.sh` runs the selected
listwise training workflow in the foreground. The separate environment setup
below is only for reproducing the project on a local machine. For inference-only
evaluation, the selected reranker can be used immediately after `download.sh`;
`run.sh` is provided to reproduce its training process.

## 1. Project layout

```text
<project root>/
├── download.sh
├── run.sh
├── retrieval_utils.py
├── retrieval.py
├── evaluate.py
├── eval_utils.py
├── requirements.txt
├── report.pdf
├── configs/
├── code/script/
└── data/
    └── train/
        ├── dialogues.jsonl
        └── products.jsonl
```

The training workflow expects the released training dialogues and products at
the two paths shown above.

## 2. Environment setup

The package versions are listed in `requirements.txt`. The workflow expects a
Linux environment with Python, PyTorch, CUDA, and an NVIDIA GPU. In the prepared
course environment, first check that the two training JSONL files are under
`data/train/`, and then download the required model files:

```bash
bash download.sh
```

For a separate local Ubuntu environment, create the project-local virtual
environment first:

```bash
bash code/script/setup_env.sh
source .venv/bin/activate
python code/script/check_environment.py
bash download.sh
```

The setup script creates `.venv` inside the project. Both `download.sh` and
`run.sh` automatically use `.venv/bin/python` when it exists.

`download.sh` creates the following model directories:

- `models/e5-base-v2`: dense candidate retriever;
- `models/bge-reranker-v2-m3`: initial checkpoint for reranker training;
- `models/selected-listwise-reranker`: submitted trained reranker used by
  `retrieval_utils.py`.

## 3. Train the submitted model

Run the complete selected training path with:

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

This is a foreground command: the terminal remains attached, training messages
are displayed directly, and `run.sh` returns the failing command's exit status
if any stage fails. Keep the terminal open until the command finishes.

`run.sh` executes these stages in order:

```bash
python code/script/check_sources.py
test -f prepared/manifest.json || python code/script/prepare_train.py
python code/script/v4_pipeline.py mine
python code/script/train_v4.py --mode listwise
python code/script/train_v4.py --mode listwise
```

The final repeated command is a completion check. If the training manifest
already exists, it validates the configuration, history, and model fingerprint
without training the model again.

### Complete training workflow

1. `check_sources.py` checks the source files and Python syntax.
2. `prepare_train.py` creates a deterministic target-product-disjoint split with
   4,343 training dialogues and 487 validation dialogues.
3. `v4_pipeline.py mine` runs BM25 and E5 over the training product collection,
   combines the rankings with weighted RRF, and stores the mined candidates.
4. `train_v4.py --mode listwise` builds groups containing one positive, four
   hard negatives, one medium negative, and one random negative.
5. The script fine-tunes `BGE-reranker-v2-m3` with LoRA and listwise
   cross-entropy. It validates after every epoch and selects the best checkpoint
   by MRR@10, followed by Recall@10.
6. The selected LoRA adapter is merged into the base reranker and written to
   `models/v4-reranker-listwise` as a standalone checkpoint.
7. The second `train_v4.py` invocation verifies the finished training artifacts.

During development, I ran the longer candidate/listwise/pairwise experiment
suite through a detached background launcher and monitored its log with `tail`.
For the submitted workflow, I changed the entrypoint to the foreground
`bash run.sh` command above and kept only the selected listwise reproduction
path. The submitted `run.sh` does not launch a background worker.

Important training outputs are:

```text
prepared/manifest.json
prepared/train.jsonl
prepared/validation.jsonl
prepared/v4/candidates.json
prepared/v4/mining.json
runs/v4/listwise/history.json
runs/v4/listwise/manifest.json
models/v4-reranker-listwise/
```

Rerunning `run.sh` reuses verified preparation, mining, resume, and completed
training artifacts. An interrupted epoch restarts from the most recently
completed epoch. On a clean workspace, the expected order is therefore:

```text
place training data -> download models -> check sources -> create split
-> mine BM25/E5 candidates -> train and validate LoRA reranker
-> merge best adapter -> verify final checkpoint
```

## 4. Run retrieval

The standard retrieval entrypoint is:

```bash
python retrieval.py \
  --dialogues "dialogues_path" \
  --products "products_path" \
  --output "output_path"
```

`retrieval_utils.py` loads `configs/inference.json` and runs the following final
pipeline:

```text
weighted phrase BM25 top 400 + E5 top 400
    -> weighted reciprocal-rank fusion
    -> protected union of 96 BM25 and 32 E5 candidates
    -> listwise-trained BGE cross-encoder reranking
    -> top 10 unique product IDs
```

The downloaded `models/selected-listwise-reranker` is used for this command, so
retrieval can run immediately after `download.sh`; reproducing training is a
separate workflow.

## 5. Main file reference

### Top-level files

- `download.sh`: downloads the pretrained and submitted model files.
- `run.sh`: reproduces the selected listwise training workflow in the foreground.
- `retrieval_utils.py`: implements BM25, E5 retrieval, RRF candidate fusion,
  protected union, and BGE reranking.
- `requirements.txt`: pinned Python dependencies.
- `report.pdf`: written assignment report.
- `README.md`: setup, training, retrieval, and file documentation.

### Configuration files

- `configs/inference.json`: final retrieval and reranking configuration.
- `configs/downloads.json`: model download destinations and trained-file URLs.
- `configs/models.lock.json`: pretrained model revisions.
- `configs/training.json`: train/validation split configuration.
- `configs/v4/candidates_selected.json`: selected BM25/E5 candidate settings.
- `configs/v4/training.json`: LoRA, optimizer, loss, batch, and early-stopping
  settings.
- `configs/v4/selected.json`: selected validation-time pipeline.

### Training and retrieval scripts

- `code/script/download_models.py`: downloads model snapshots and trained files.
- `code/script/check_environment.py`: prints the Python, PyTorch, CUDA, and GPU setup.
- `code/script/check_sources.py`: checks source integrity and Python syntax.
- `code/script/common.py`: shared JSON, hashing, data, metric, and retrieval helpers.
- `code/script/prepare_train.py`: creates the train/validation split and manifest.
- `code/script/v4_common.py`: defines V4 candidate variants, negative sampling, and
  ranking losses.
- `code/script/v4_pipeline.py`: runs candidate selection, mining, reranker comparison,
  and experiment summaries.
- `code/script/train_v4.py`: trains, resumes, validates, selects, merges, and exports
  the V4 LoRA reranker.
- `code/script/run_retrieval.py`: runs retrieval with an explicit configuration file.
- `code/script/exact_metrics.py`: calculates exact Recall@10 and MRR@10.
- `code/script/public_evaluation.py`: runs the recorded public retrieval/evaluation
  workflow.

### Report and supporting scripts

- `code/script/generate_report_examples.py`: produces the five-sample, three-prompt
  Qwen generation experiment.
- `code/script/generation_analysis.py`: summarizes the saved generation outputs.
- `code/script/build_report_v4.py`: builds the V4 report artifacts.
- `code/script/run_ablations.py` and `code/script/v3_experiments.py`: reproduce the
  preliminary retrieval and reranking comparisons.
- `code/script/test_v4.py`, `code/script/test_v4_resume.py`, and
  `code/script/test_ta_retrieval_flow.py`: test the V4 pipeline, resume behavior, and
  retrieval entrypoint.
- `code/script/verify_submission_archive.py`: validates the final archive layout and
  internal checksums.

The remaining scripts in `code/script/` are earlier experiment runners, diagnostics,
report builders, or packaging helpers retained with the code used during the
assignment.

## 6. Quick command summary

```bash
# Optional local environment
bash code/script/setup_env.sh
source .venv/bin/activate

# Download all models used by the selected pipeline
bash download.sh

# Reproduce the selected reranker
CUDA_VISIBLE_DEVICES=0 bash run.sh

# Run final retrieval
python retrieval.py --dialogues dialogues.jsonl \
  --products products.jsonl --output predictions.json
```
