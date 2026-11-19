# v12 evaluation evidence

These JSON files are outputs of the provided `eval.py score` on the PUBLIC
evaluation sets. They contain aggregate metrics, not training targets or model
predictions. The four `checkpoint-*_subset.json` files all score the same
600-step selection subset and are read by `code/build_v12_curve.py` for the
report learning curve. `v12_validation_loss.csv` contains 21 held-out loss
measurements exported from the original Trainer state by
`code/export_v12_history.py`. These losses are plotted as losses, without
conversion into checkpoint accuracies. `v12_user_reported_accuracy.csv` stores the eight values supplied by the
student for steps 6,000 through 10,200 at intervals of 600 steps. Their original score JSONs,
subset sizes, and identity with the 600-step subset have not been verified.
The report text and source CSV distinguish these from JSON-backed observations;
unknown fields remain empty rather than being inferred. The `tools_full` and `negotiation_full` files are the
complete official PUBLIC scores for the selected checkpoint-10800 adapter.
`fresh_v12_20261026_124322_error_analysis.json` contains only aggregate
diagnostics and supports the reported 2,165 exact must-call steps.
`untuned_base_first100_128.json` and `v12_first100_128.json` are official
scores on the same first 100 PUBLIC steps under the same 128-token generation
cap. `measure_base_first100.sh` reproduces this Q5 comparison. The older
`v12_first100.json` is a separate submitted-adapter smoke test with the
default generation cap.
