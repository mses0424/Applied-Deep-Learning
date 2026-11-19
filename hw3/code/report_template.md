# ADL 2026 HW3 — Tool-Calling Training (v12)

Student ID: **{{STUDENT_ID}}**  
{{STUDENT_NAME_LINE}}

## Q1. Data

I built a joint tool-calling and buyer dataset with 49,349 training conversations and 2,218 held-out conversations. Before the 3,072-token length filter, I counted 106,816 supervised assistant decisions in training and 5,143 in validation. I kept each source conversation in one split to prevent its turns from appearing in both. I converted every row to Qwen-compatible `messages` plus declared `tools`, with tool names and JSON arguments in assistant `tool_calls`. I used the homework PUBLIC files only for evaluation and a discard-only text-overlap check; I never used their records, answers, or predictions as training targets.

- I generated 7,408 stateful policy conversations covering service rules, clarification, denial, and stopping.
- I generated 5,760 policy-branch rows that use an observed prerequisite result in a later call.
- I generated 5,000 BFCL document-based cases with typed calls, dependencies, and distractors.
- I converted 7,359 BFCL executed multi-turn cases with simulator results before subsequent decisions.
- I converted 3,636 Toucan trajectories with varied multi-turn tool schemas.
- I mixed 19,386 APIGen-MT, Synth-APIGen, When2Call, buyer, and related targeted rows.
- I retained 800 ToolACE rows to practice continuation after a tool result.

I used the public training sources [BFCL](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard), [Salesforce APIGen-MT-5k](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) through its [parsed mirror](https://huggingface.co/datasets/minpeter/apigen-mt-5k-parsed), [Argilla Synth-APIGen-v0.1](https://huggingface.co/datasets/argilla/Synth-APIGen-v0.1), [NVIDIA When2Call](https://huggingface.co/datasets/nvidia/When2Call), [Team-ACE ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE), and [Agent-Ark Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M). I used [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) only as a teacher to synthesize one training subset. In the included preparation code, I normalized each source into tool schemas and assistant decisions, rejected malformed arguments, and preserved source groups across the split. For BFCL, I replayed public multi-turn answers in its simulator so the next decision sees an observed result. I kept 796 of 800 source episodes, rejected 3 exact PUBLIC overlaps and 1 inconsistent replay, and found zero exact or near matches among 92,913 substantive user messages in the final mix. I also synthesized 6,144 policy conversations across 48 service domains and four rule families; 5,760 entered training.

## Q2. Training method

I kept the required [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) transformer weights frozen and used `Qwen/Qwen3-0.6B` only for the official chat tokenizer and template. For each assistant decision, I rendered the visible history and declared tools with `enable_thinking=False` and an assistant generation prompt. I assigned label `-100` to every prompt token and cross-entropy loss to the assistant target, including the final end-of-message token. For rows marked `loss_on_last_assistant_only`, I supervised only the last decision while retaining the earlier conversation as context. Where a row allowed it, I removed older completed user tasks from an overlong history; I discarded examples still exceeding 3,072 tokens rather than truncating the supervised target. I trained the 25 chat/tool token rows (IDs 151644–151668) through PEFT `trainable_token_indices` together with LoRA.

I shuffled and mixed 43,449 general tool and 5,900 buyer training rows in the same run. I taught calls, post-tool continuation, no-call responses, `wait`, and `order` to one adapter; I did not train or load a separate buyer model. At inference, I merged this adapter into the required Base model and used the supplied `eval.py` with greedy decoding and the official scorer.

## Q3. Hyperparameters

- I set LoRA rank 128, alpha 192, and dropout 0.05 to cover varied schemas while submitting only an adapter.
- I adapted q, k, v, o, gate, up, and down projections so attention and MLP layers could both change.
- I used learning rate 1.2e-4, cosine decay, 4% warmup, AdamW, weight decay 0.01, and gradient clipping at 1.0.
- I trained for 2 epochs with a 3,072-token maximum to include multi-turn histories that fit GPU memory.
- I used microbatch 1 per GPU and accumulation 8 on two RTX 2080 Ti cards, giving effective batch 16.
- I used FP16 distributed training, which is supported on the available Turing GPUs, with seed 3407.

I trained in a Python 3.11.15 environment with torch 2.11.0, transformers 5.13.1, PEFT 0.19.1, vLLM 0.25.1, and Jinja2 3.1.6. I selected checkpoint 10,800 because it scored best among the retained checkpoints on one fixed PUBLIC subset, even though checkpoint 12,000 had lower held-out loss. I then ran the full official PUBLIC evaluations on the selected checkpoint.

## Q4. Learning curve

![v12 checkpoint accuracy and recorded validation loss](../results/v12_learning_curve.svg)

I plotted twelve v12 checkpoint scores using the same markers and solid connecting lines. At steps 6,000, 6,600, 7,200, 7,800, 8,400, 9,000, 9,600, and 10,200, I include user-provided accuracies of 65.12%, 65.43%, 66.08%, 66.36%, 65.96%, 67.12%, 66.89%, and 67.35%, respectively; I have not verified their original scoring files or evaluation-subset identity. At steps 10,800, 11,400, 12,000, and 12,284, the retained official JSONs give 68.04%, 65.96%, 65.96%, and 65.96% weighted accuracy on the same fixed 600-step PUBLIC subset. I used PUBLIC only for evaluation and checkpoint selection, never for gradients.

I also plotted 21 actual held-out validation-loss measurements from step 600 to 12,284. The loss decreased from 0.2288 to 0.1930, while the late PUBLIC subset accuracy declined. Loss and step accuracy measure different behavior; I did not convert loss or other training runs into v12 accuracy. Q5 gives the separate evaluation on the complete 13,021-step PUBLIC set.

## Q5. Performance

I evaluated v12 checkpoint 10,800 on all 13,021 PUBLIC Part 1 steps using the supplied `eval.py score` (lenient parser, first-step rule). I measured **{{TOOL_SCORE_PRECISE}}** weighted step accuracy (**{{TOOL_SCORE}}** rounded) and 72.58% ordinary accuracy. This full-set score is separate from Q4's 600-step subset. Of 3,254 must-call steps, 2,165 matched exactly; 604 used the wrong tool and 353 used the correct tool with wrong arguments. For the untuned Base comparison, I scored both models on the same first 100 PUBLIC steps with a shared 128-token cap: Base 0.52%, v12 68.04%. Since 98 Base replies hit that cap, I interpret this comparison only under that cap, not as the Base's full-set score. My complete PUBLIC weighted accuracy remains below 70%; PRIVATE performance is unknown.

## Q6. Buyer agent

I included 3,700 balanced order rows, 1,100 high-offer wait rows, and 1,100 wrong-offer wait rows among 5,900 buyer training rows. I trained the buyer to keep its private budget and target undisclosed, ask for a lower price with `wait`, and call `order` only when the latest formal offer matches the product, unit price, quantity, and quoted evidence. I used the following actual synthetic training conversation (`negotiation-balanced-48`); the private target was $39.84 per unit for three units of SKU-I, and the total budget was $173.70:

```text
Buyer: Could you offer 3 of SKU-I near $32.67 each?
Tool: wait({})
Seller [deal-48-2]: Offer: SKU-I at $39.84 per unit, qty 3.
                    This is my current formal offer.
Buyer: Agreed. I accept this current offer.
```

I paired that final buyer reply with this `order` call, whose arguments quote the seller's standing offer:

```json
{
  "name": "order",
  "arguments": {
    "product_id": "SKU-I", "agree_price": 39.84, "qty": 3,
    "evidence_message_ids": ["deal-48-2"],
    "evidence_quote": "Offer: SKU-I at $39.84 per unit, qty 3."
  }
}
```

I evaluated all 2,607 PUBLIC negotiations with the official scorer. I measured **{{ORDER_ACCURACY}}** order accuracy, **{{DEAL_VOLUME}}** deal volume (fill rate), and **{{DEAL_PRICE}}** relative deal price, where lower is better. I report buyer response accuracy as **{{RESPONSE_ACCURACY}}** because I could not run the required Qwen3-8B response judge on my 11 GB GPU. I chose to wait on high or mismatched offers to protect price and validity; more waiting can lose deals, while accepting earlier can increase volume at a worse price. I therefore report both price and volume rather than treating either alone as the whole buyer result.
