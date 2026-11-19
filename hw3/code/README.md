# Code inventory for the submitted v12 adapter

The selected model is the single LoRA adapter in `../adapter_checkpoint/`. Part 1 and the Part 2 buyer use the same weights. The root `README.md` gives the required step-by-step environment, data preparation, training, and evaluation commands. This file explains each source file included in `code/`.

## Environment and data sources

| File | Purpose |
|---|---|
| `create_ta_venv.sh` | Create the Python environment and install the evaluation and training dependencies. |
| `requirements-train.txt` | Record extra training and data-processing packages. |
| `check_environment.py` | Check Python, packages, GPU, and optional source-data availability. |
| `download_v12_sources.py` | Download the public training sources used by v12 at pinned revisions and check their hashes. |
| `download_public_v10.py` | Download pinned BFCL and Toucan training files reused in the v12 data mix. |
| `public_guard_v10.py` | Reject candidate training rows that overlap homework PUBLIC text; it does not export PUBLIC examples or labels. |
| `audit_data.py` | Check conversation format, splits, and dataset statistics. |

## Training-data construction

`prepare_v12_data.sh` is the entry point. It calls earlier converters and generators because the v12 mixture reuses their output; versioned filenames do not indicate additional submitted adapters.

| File | Purpose |
|---|---|
| `prepare_v12_data.sh` | Assemble and audit the final v12 training and validation JSONL files. |
| `prepare_training_data.sh` | Build initial synthetic tool and buyer examples needed by later preparation steps. |
| `generate_data.py` | Generate original tool-call and buyer conversations in Qwen/OpenAI message format. |
| `generate_dependent.py` | Generate multi-tool examples whose later calls depend on earlier tool results. |
| `generate_multistep.py` | Generate multi-step continuation examples from synthetic conversations. |
| `generate_policy_v10.py` | Generate policy-driven tool-use conversations from original rules. |
| `generate_policy_v11.py` | Generate additional policy and state-change conversations. |
| `generate_policy_branch_v12.py` | Generate policy-branch examples with observable prerequisites. |
| `synthesize_bfcl_docs.py` | Create new requests from BFCL function documentation without copying BFCL answers. |
| `prepare_bfcl_stateful_v9.py` | Build stateful tool examples from BFCL function documentation. |
| `prepare_bfcl_official_v10.py` | Convert public BFCL training tasks and possible answers into tool-call turns. |
| `prepare_bfcl_executed_v12.py` | Replay public BFCL training trajectories with the official simulator before conversion. |
| `prepare_toucan_v10.py` | Convert public Toucan training trajectories to the message format used for training. |
| `prepare_toolace_data.py` | Convert public ToolACE training dialogues to ordered tool-call turns. |
| `prepare_corrective_data.py` | Build filtered corrective tool and buyer examples. |
| `prepare_targeted_data.py` | Build filtered examples for difficult tool-use cases. |
| `prepare_v6_calibration.py` | Add a small calibration set after the ToolACE conversion. |
| `prepare_fresh_v11.py` | Mix public and original data, assign train/validation splits, and write the final curriculum. |

## Training and evaluation

| File | Purpose |
|---|---|
| `train.py` | Fine-tune the frozen Qwen3-0.6B-Base with LoRA and assistant-token loss masking. |
| `run_v12_training.sh` | Train, score retained checkpoints with the staff evaluator, select an adapter, and evaluate both PUBLIC parts. |
| `launch_fresh_v12.sh` | Start the measured training workflow in a detached `tmux` session. |
| `validate_adapter.py` | Check that the adapter has the expected base model and required PEFT files. |
| `build_stratified_ids.py` | Make an evaluation-only ID list for a fixed PUBLIC subset without copying PUBLIC content. |
| `analyze_public_errors.py` | Summarize aggregate errors from evaluation predictions without exporting examples. |
| `measure_base_first100.sh` | Run the report's like-for-like first-100 comparison against the untuned Base model. |

The supplied `eval.py`, `eval/` package, and PUBLIC files are part of the course pack, not this submission. PUBLIC data is used for overlap rejection and evaluation only, never as supervised training targets.

## Report and packaging

| File | Purpose |
|---|---|
| `report_template.md` | Source text for the six required report answers. |
| `export_v12_history.py` | Export recorded validation losses from the original Trainer state to the included evidence CSV. |
| `build_v12_curve.py` | Export official JSON-backed and user-reported v12 checkpoint scores, retaining source labels. |
| `plot_curve.py` | Plot checkpoint scores with uniform markers and recorded validation loss as an SVG; sources remain in the CSV and report text. |
| `finalize_report.py` | Insert the student information and measured scores into the report source. |
| `build_report_pdf.py` | Render the final report PDF. |
| `evidence/` | Store aggregate score JSONs, manifests, and the evidence README used for the report. |
| `finish_submission.sh` | Preserve the final PDF and package the adapter and evidence; set REBUILD_REPORT=1 to regenerate the source-draft PDF. |
| `package_submission.py` | Create the required student-ID ZIP layout with the adapter files, code, README, and report. |

The model weights to submit are `../adapter_checkpoint/adapter_config.json` and `../adapter_checkpoint/adapter_model.safetensors` inside the student-ID ZIP. No cloud-hosted model or full base-model weights are part of the submission.
