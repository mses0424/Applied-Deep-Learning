# HW2 report experiment handoff

This file is an evidence index, not a finished report.  Write the interpretation
in your own words and verify every claim against the linked JSON/CSV artifacts.

## Completion status

- Q1 train/validation comparisons: complete.
- Q1 listwise and pairwise training runs: complete; both stopped after epoch 3.
- Full cold-cache public retrieval and official judge: complete.
- Q2 generation: complete, with 5 distinct samples and 3 prompts per sample.
- Q2 qualitative analysis: intentionally left for the student.
- Private performance: unknown and must not be claimed.

## Q1 experiment inventory

There are three related groups of experiments.  Avoid reporting one ambiguous
total without explaining what is counted.

1. Eleven v3 pilot configurations, E00--E10, tested lexical, dense, fusion,
   product/query views, pretrained cross-encoders, and random versus hard-negative
   LoRA.  `B40` and `B120` are marked `NOT RUN` and must not be described as completed.
2. Five controlled v4 candidate-retrieval variants, R1--R5.
3. Seven final reranking configurations: L0, C0, C1, listwise, listwise+blend,
   pairwise, and pairwise+blend.

Primary raw tables:

- `results/v3/ablations.csv`
- `results/v4/comparisons.csv`
- `results/v4/candidate_selection.json`
- `results/v4/selection.json`

### V4 candidate retrieval (487 target-disjoint validation dialogues)

| ID | Controlled change | Recall@128 | Exact Recall@10 | Exact MRR@10 |
|---|---|---:|---:|---:|
| R1 | lexical only within the 128-candidate budget | 0.798768 | 0.650924 | 0.436197 |
| R2 | weighted BM25/E5 RRF | 0.827515 | 0.650924 | 0.436197 |
| R3 | protected 96 BM25 + 32 BGE union | **0.837782** | 0.650924 | 0.436197 |
| R4 | protected 96 BM25 + 32 E5 union | 0.835729 | **0.657084** | **0.456291** |
| R5 | R4 with user-turn-only E5 query | **0.837782** | **0.657084** | 0.447630 |

R5 was selected by the declared rule: Recall@128 first, then MRR@10, then time.
R4's better top-10 MRR is therefore not inconsistent with selecting R5 for
candidate mining.  Explain this distinction rather than claiming R5 won every metric.

### V4 reranking comparison

| ID | Method | Candidate Recall | Exact Recall@10 | Exact MRR@10 |
|---|---|---:|---:|---:|
| L0 | exact previous v3 method | 0.761807 | 0.731006 | 0.625991 |
| C0 | pretrained V2-M3 on v4 candidates | 0.837782 | 0.509240 | 0.320382 |
| C1 | v3 hard-negative model on v4 candidates | 0.837782 | 0.763860 | 0.649895 |
| T_listwise | v4 listwise LoRA | 0.837782 | **0.796715** | **0.689798** |
| T_listwise_blend | listwise plus retrieval-rank blend | 0.837782 | 0.794661 | 0.655961 |
| T_pairwise | listwise plus pairwise loss | 0.837782 | 0.794661 | 0.688386 |
| T_pairwise_blend | pairwise plus retrieval-rank blend | 0.837782 | 0.792608 | 0.657550 |

T_listwise is the selected configuration.  The blend variants and the additional
pairwise term did not improve the selection metric.  Do not attribute the entire
v3-to-v4 gain to one isolated change because candidate budget, E5 retrieval,
sequence length, sampling curriculum, batch accumulation, and training settings
changed together.

### Training curve evidence

| Loss | Epoch | Recall@10 | MRR@10 | Fixed-train probe top-1 | Fixed-train NLL |
|---|---:|---:|---:|---:|---:|
| listwise | 0 | 0.509240 | 0.320197 | 0.53125 | 1.304103 |
| listwise | 1 | **0.796715** | **0.688772** | 0.90625 | 0.346208 |
| listwise | 2 | 0.786448 | 0.680412 | 0.953125 | 0.241013 |
| listwise | 3 | 0.782341 | 0.683900 | 0.953125 | 0.248777 |
| pairwise | 0 | 0.509240 | 0.320197 | 0.53125 | 1.304103 |
| pairwise | 1 | **0.794661** | **0.687359** | 0.90625 | 0.345349 |
| pairwise | 2 | 0.786448 | 0.676265 | 0.953125 | 0.235965 |
| pairwise | 3 | 0.782341 | 0.677882 | 0.953125 | 0.246808 |

Epoch 1 was selected for both losses.  The later fixed-train probe improvement
alongside weaker validation metrics is evidence worth discussing as possible
overfitting; it is not proof of the cause.

### Final public measurement

| Measurement | Value |
|---|---:|
| Queries | 1,000 |
| Graded Recall@10 | **0.968** |
| Graded MRR@10 | **0.880969** |
| Judge parse failures | 0 |
| Cold retrieval wall time | 1,473.4 s (24 min 33 s) |
| Peak CUDA reserved in retrieval | 1.14 GiB |

Artifacts:

- `results/v4/public.json`
- `results/v4/public.run.json`
- `results/v4/evaluation.json`
- `results/v4/evaluation.receipt.json`

Public passing does not establish private passing.  The cold timing was measured
on an RTX 2080 Ti, not the TA's RTX 3070.

## Q2 generation experiment

The exact experiment is in `results/v4/generations.json`.  It records the model
fingerprint, prompts, system prompt, supplied conversation/product evidence,
outputs, token counts, EOS status, and truncation flags.  Samples were selected
by the lowest SHA256 of `2026:<qid>`, independently of gold labels or output quality.

| Sample | QID | Retrieved PID | Direct | Constraints | Evidence |
|---:|---|---|---:|---:|---:|
| 1 | public-test-home-and-kitchen-000019 | B07PNKVHD4 | 97 tokens | 172 | 136 |
| 2 | public-test-sports-and-outdoors-000055 | B0081SRG2I | 127 | 109 | 173 |
| 3 | public-test-sports-and-outdoors-000034 | B016CKO30Y | 100 | 136 | **256, truncated** |
| 4 | public-test-beauty-000156 | B01H19L74K | 148 | 133 | 116 |
| 5 | public-test-sports-and-outdoors-000046 | B01MDLUWZ7 | 105 | 116 | 127 |

No conversation or product evidence was truncated before generation.  One model
answer, sample 3's evidence prompt, hit the 256-token output limit.  This is a
real negative result and should not be hidden.

When writing the qualitative section, check at least the following directly in
the JSON rather than assuming the prompt worked:

- Whether every claimed match is supported by the supplied catalogue text.
- Whether text presented as a quotation is actually verbatim evidence.
- Whether all firm user constraints are covered.
- Whether `unspecified` is used correctly rather than merely repeating a known requirement.
- Sample 3's truncated evidence answer and whether its quoted dimensions support the recommendation.
- Sample 5's evidence wording versus the exact catalogue wording.
- Retrieval error versus generation error: all analysis must use the PID that was actually supplied.

## Files to preserve in the submission

- `scripts/generate_report_examples.py` for Q2 reproduction.
- All `scripts/train_v4.py`, `scripts/v4_pipeline.py`, and shared helpers used by Q1.
- `configs/v4/`, `runs/v4/` evidence (without large resumable weights if the packager excludes them),
  and `results/v4/` receipts/tables.
- `archive/v3_20261007/` for the reported v3 comparisons.
- The final student-written report source and generated PDF.

Do not include statements that B40/B120 ran, that public data trained the model,
that private passed, or that TA hardware timing was directly measured.
