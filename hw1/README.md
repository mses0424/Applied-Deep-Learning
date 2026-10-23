# ADL HW1 — Bilingual Multi-label User State Prediction

Student ID: `r15922139`

The submitted system is a dual-seed XLM-RoBERTa ensemble combined with
character/word TF-IDF LinearSVC models, selective recent-context features, and
Chinese/English calibration. All model selection uses `train.jsonl` only; no
test labels are used.

## 1. Official inference

The grading environment provides Python 3.10 and the packages listed in the
assignment. From the submission directory, run exactly:

```bash
bash ./download.sh
bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
```

`download.sh` downloads and extracts the model into `model/`. `run.sh` requires
exactly three arguments: context JSON, test JSON, and output CSV. Inference is
offline after the download and creates the output directory when necessary. The
serialized TF-IDF/LinearSVC artifacts require `scikit-learn==1.7.2`; `run.sh`
checks this version before inference and stops with an actionable error if the
local environment is incompatible.

Optional output-format check:

```bash
python3 validate_submission.py /path/to/test.json /path/to/pred/prediction.csv
```

## 2. Training environment

Training data is not included in the submission. Place the official
`train.jsonl` in the submission directory. Create an isolated environment:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

The completed experiment used two NVIDIA RTX 2080 Ti GPUs, mixed precision,
five-fold OOF validation, and approximately 7–9 hours. The submitted training
entry point launches two distributed processes and therefore expects two visible
CUDA GPUs.

Verify the environment before training:

```bash
.venv/bin/python -c "import torch, transformers, sklearn; print(torch.__version__, transformers.__version__, sklearn.__version__); print(torch.cuda.is_available(), torch.cuda.device_count())"
```

The last line should report `True 2` (or more than two available GPUs).

## 3. Reproduce the final model

Run the complete pipeline:

```bash
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
CUDA_VISIBLE_DEVICES=0,1 bash train_all.sh
```

The final Transformer configuration is:

| Setting | Value |
|---|---|
| Backbone | `FacebookAI/xlm-roberta-base` |
| OOF folds | 5 |
| Epochs | 6 |
| Seeds | 42, 3407 |
| Maximum length | 512 tokens |
| Batch size | 2 per GPU |
| Gradient accumulation | 4 |
| Backbone learning rate | `1.5e-5` |
| Classification-head learning rate | `8e-5` |
| Mixed precision | fp16 |

The script performs the following steps:

1. Fine-tunes `FacebookAI/xlm-roberta-base` for six epochs with seeds 42 and
   3407 using five-fold OOF validation.
2. Trains two final full-data Transformer models and averages their prediction
   probabilities.
3. Trains ten TF-IDF LinearSVC classifiers and selects `C`, Transformer/linear
   weights, and decision thresholds independently for each label using OOF
   predictions.
4. Evaluates selective recent-context LinearSVC features for weak labels.
5. Evaluates a nested-OOF logistic stacker and retains it only if it improves
   held-out F1. It was rejected for the final XLM-R system.
6. Applies separate Chinese/English calibration and saves the final inference
   configuration.

Training outputs are written to:

```text
artifacts/xlmr_final_model/
```

Completed folds and seed models are checkpointed. Re-running the same command
continues from completed checkpoints. To force a new run without overwriting an
existing model, use a new output directory:

```bash
MODEL_DIR="$PWD/artifacts/xlmr_new_run" \
CUDA_VISIBLE_DEVICES=0,1 \
bash train_all.sh
```

If SSH disconnects, the foreground command stops unless it is launched from
`tmux`, `screen`, or with `nohup`. A resumable background example is:

```bash
MODEL_DIR="$PWD/artifacts/xlmr_final_model" \
CUDA_VISIBLE_DEVICES=0,1 \
nohup bash train_all.sh > training.log 2>&1 &

tail -f training.log
```

Running the same command again with the same `MODEL_DIR` reuses completed
fold/seed checkpoints. Do not change the backbone, fold count, seed list, or
training hyperparameters when resuming.

Display the saved OOF metrics with:

```bash
.venv/bin/python show_metrics.py --model-dir artifacts/xlmr_final_model
```

Successful training must produce at least the following files:

```text
artifacts/xlmr_final_model/inference_config.json
artifacts/xlmr_final_model/model_seed_42/model.safetensors
artifacts/xlmr_final_model/model_seed_3407/model.safetensors
artifacts/xlmr_final_model/linear_model.joblib
artifacts/xlmr_final_model/context_linear_model.joblib
artifacts/xlmr_final_model/training_metrics.json
```

Before packaging, run a local inference and validate its structure:

```bash
bash run.sh /path/to/context.json /path/to/test.json /tmp/prediction.csv
.venv/bin/python validate_submission.py /path/to/test.json /tmp/prediction.csv
```

The measured train-only OOF results were:

| System | Macro-F1 | Micro-F1 |
|---|---:|---:|
| Dual-seed XLM-R | 0.8196 | 0.8257 |
| XLM-R + LinearSVC | 0.8308 | 0.8356 |
| Final ensemble | 0.8358 | 0.8413 |

These are training-data OOF validation results, not hidden-test scores.

## 4. Reproduce the controlled experiments

The following command reproduces the fixed-split TF-IDF/context/long-input
ablations and the matched mDeBERTa/XLM-R Chinese-English comparison:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_report_experiments.sh
```

Measured JSON outputs are saved under `reports/`. This command only runs the
experiments; it does not generate or modify the submitted PDF report.

## 5. Package a newly trained model

After training, create the downloadable archive:

```bash
bash package_model.sh
```

This produces `model.tar.gz` with a top-level `model/` directory. Upload the
archive to a publicly readable location and update the file ID and SHA-256 in
`download.sh`. Models, datasets, `.venv`, and raw artifacts must not be placed
inside the NTU COOL submission ZIP.

## Main files

- `download.sh`: downloads, verifies, and extracts the submitted model.
- `run.sh`, `predict.py`: required offline inference pipeline.
- `train_all.sh`: complete two-GPU training entry point.
- `train_transformer.py`: Transformer OOF and final full-data training.
- `train_linear.py`: TF-IDF LinearSVC training and per-label blending.
- `train_context_linear.py`: selective recent-context component.
- `train_stacker.py`: nested-OOF stacker evaluation.
- `finalize_ensemble.py`, `optimize_language_calibration.py`: final selection
  and calibration.
- `report_ablations.py`, `experiment_transformer_split.py`: controlled
  experiments reported in `report.pdf`.
- `validate_submission.py`: prediction-format validation without gold labels.
