# Joint trajectory–intention model audit

**Scope.** Read-only audit of the saved 15-step JAAD experiments and the implementation in `src/models/joint_transformer_gate.py` and `scripts/train_joint_transformer_gate.py`. No model was trained or updated. The diagnostic script performs forward passes and `autograd.grad` only; it creates no optimizer and calls no `optimizer.step`.

**Compared checkpoints.** Independent scene-aware trajectory Transformer, seed 123: test pixel ADE **11.0585** (`results/trajectory_transformer_scene_15x15_seed123/metrics.json`). Joint Transformer gate, seed 123: test pixel ADE **25.0303** (`results/joint_transformer_gate_15x15_seed123/metrics.json`). Both use the same `jaad_sequences_scene_15x15` data root and 15-step horizon. The joint ADE is 2.26× the independent model's. The joint result is one training seed, so this audit diagnoses this run/configuration rather than estimating seed variance.

## 1. Architecture comparison

| Component | Trajectory-only Transformer | Joint trajectory–intention Transformer |
|---|---|---|
| Encoder | 8-D target history → linear projection → 3-layer, 128-D Transformer; separate 512-D scene MLP. | Same-width/depth target Transformer and scene MLP, plus a neighbor GRU and proposal-fusion path. |
| Trajectory decoder | MLP maps concatenated `[target_context, scene_context]` to 15 × 2 future coordinates. | MLP maps `[fused_context, target_context]` to 15 × 2. `fused_context` already mixes target, scene, and gated social features. |
| Intent head | None. | Main intent logit from `fused_context`, plus an auxiliary proposal-intent logit used by the gate and auxiliary BCE. |
| Feature sharing | Trajectory-only objective; no intent task shares its encoder. | Main intent, proposal-intent, and trajectory losses all backpropagate through substantial shared target/scene/social/fusion representations. |

The model definitions are [trajectory-only](../src/models/trajectory_transformer.py#L7) and [joint](../src/models/joint_transformer_gate.py#L7). The joint trajectory head therefore does not simply add an intent classifier beside the old trajectory model: it changes the trajectory decoder input from target+scene to fused+target and lets intention objectives update shared features.

There is also a transfer-initialization hazard for the separate `*_pretrained_*` run. The loader maps the trajectory-only `decoder.*` weights directly onto `traj_head.*` when tensor shapes match ([training script](../scripts/train_joint_transformer_gate.py#L143)). But the source decoder's first input halves mean `[target_context, scene_context]`, while the joint decoder receives `[fused_context, target_context]`. Same shape does not mean same feature semantics; this transfer is not a valid direct decoder warm start. The pretrained variant's test ADE is 28.7904, worse than the non-pretrained joint run, though that alone does not isolate this bug.

## 2. Loss audit

For the standard joint run, the checkpoint arguments specify fixed `prior_weight=0.5`, `traj_weight (λ)=1.0`, and `ambiguous_weight=0.2` for every epoch. The actual training loop optimizes:

```text
L = BCE(main_intent)
  + 0.5 × BCE(proposal_intent)
  + 1.0 × SmoothL1(future_xy_normalized)
  + 0.2 × 0.5 × (proposal_logit² + main_intent_logit²) on ambiguous samples
```

The independent trajectory-only model optimizes SmoothL1 alone and selects the checkpoint by lowest validation **pixel ADE** ([trainer](../scripts/train_trajectory_transformer.py#L134)). The joint trainer selects by `intent_auc + 0.1 × intent_f1 − 0.01 × trajectory_ade_pixel`, not trajectory ADE alone ([joint trainer](../scripts/train_joint_transformer_gate.py#L190)). For this standard run the selected epoch 14 also happens to have the lowest validation pixel ADE, so selection-score mismatch is a protocol difference, not by itself the explanation for its poor test ADE.

The joint script records only the **combined validation loss**, not per-epoch intent/prior/trajectory components or training component losses. Its loop saves only the current best-scoring checkpoint, not epoch snapshots ([logging/checkpoint code](../scripts/train_joint_transformer_gate.py#L190)). Thus exact per-component losses and gradient cosines for epochs 1, 5, and 10 cannot be recovered from current artifacts without retraining, which was explicitly not done.

| Epoch/checkpoint | Joint combined validation loss | Joint val pixel ADE | Intent / trajectory component losses for that epoch |
|---:|---:|---:|---|
| 1 | 0.5992 | 44.20 | Not logged / epoch checkpoint absent |
| 5 | 0.4955 | 32.94 | Not logged / epoch checkpoint absent |
| 10 | 0.6903 | 32.47 | Not logged / epoch checkpoint absent |
| 14 (selected) | 0.7066 | 21.73 | Recomputed below at saved weights, on fixed diagnostic subsets |

At the saved epoch-14 weights, the read-only diagnostic measured:

| Diagnostic subset (512 main examples) | Main intent BCE | Weighted proposal BCE | Weighted ambiguity term | Weighted trajectory SmoothL1 (`λ=1`) | Intent / trajectory loss ratio |
|---|---:|---:|---:|---:|---:|
| Class-balanced train subset | 0.00524 | 0.00436 | 0.00142 | 0.0000914 | 121× |
| Class-balanced validation subset | 1.62170 | 0.72958 | 0.00148 | 0.0001667 | 14,117× |

The validation subset is intentionally class-balanced to match the joint trainer's weighted sampler; it is not the natural-prevalence validation aggregate. The large train/validation intent-loss gap at the selected weights is consistent with severe overfit/confidence shift. More importantly, on shared parameters the trajectory gradient is tiny relative to the intention gradient (quantified below). A separate existing ablation increased `λ` to 5, but test ADE only moved from 25.0303 to 24.7082; that change alone did not approach 11 px (`results/joint_transformer_gate_15x15_traj5_seed123/metrics.json`).

**Implementation caveat:** `run_epoch()` overwrites the supervised loss if called with nonzero `ambiguous_weight` ([lines 75–81](../scripts/train_joint_transformer_gate.py#L75)). The current `main()` training loop does not use that helper for training; it adds the ambiguity regularizer correctly, and validation calls the helper with its default weight 0. This is a latent helper bug, not the cause of the saved standard run.

## 3. Gradient conflict analysis

The requested epoch 1/5/10 cosine values are **unavailable**: only the selected epoch-14 checkpoint exists. The added [`scripts/audit_joint_model_gradients.py`](../scripts/audit_joint_model_gradients.py) accepts per-epoch checkpoints if they are ever available; it will not synthesize them by retraining.

Instead, I computed task gradients at the existing selected epoch-14 checkpoint. “Intent” here is the actual combined intention objective (main BCE + weighted proposal BCE + ambiguous-logit regularizer); “trajectory” is weighted SmoothL1. Cosine/norms are over 830,530 parameters in the shared target, scene, social, proposal, gate, and fusion modules; task-specific output heads are excluded.

| Fixed diagnostic subset | Cosine(intent, trajectory) | `||λ∇Ltraj|| / ||∇Lintent||` | Interpretation |
|---|---:|---:|---|
| Class-balanced train, 512 examples | +0.194 | 0.00232 | Weakly aligned here; trajectory gradient ≈ 1/432 of intent gradient. |
| Class-balanced validation, 512 examples | **−0.322** | **0.0000739** | Conflicting directions on this held-out subset; trajectory gradient ≈ 1/13,540 of intent gradient. |

This is evidence of **validation-time gradient conflict and severe task-gradient scale imbalance at the saved checkpoint**, but not proof that cosine was negative throughout training: one checkpoint and two fixed subsets cannot replace epoch 1/5/10 snapshots. The train/validation sign flip also points to representation/generalization shift, not a uniformly conflicting training signal.

## 4. Representation analysis

On the same 2,048 evenly spaced test examples, I compared the target-encoder output and the actual 256-D input to each trajectory head. Variance is the mean per-dimension population variance across examples.

| Representation | Width | Mean sample L2 norm (± SD) | Mean feature value | Mean per-dimension variance |
|---|---:|---:|---:|---:|
| Trajectory-only target encoder | 128 | 20.630 ± 0.383 | −0.0910 | 0.0662 |
| Joint target encoder | 128 | 24.184 ± 5.880 | −0.0633 | **3.9449** |
| Trajectory-only decoder input `[target, scene]` | 256 | 21.305 ± 0.370 | 0.0714 | 0.0347 |
| Joint trajectory decoder input `[fused, target]` | 256 | 34.021 ± 9.597 | 0.5240 | **3.0537** |

The joint representations vary far more across test examples. This supports a scale/stability problem in the shared representation, but feature variance alone does not establish causality; it should be read together with the gradient and loss measurements.

## 5. Ablation design proposal (not implemented)

No architecture change or ablation was implemented in this audit. Keep the current parameter count fixed for the first checks.

| Option | Role / proposal | Priority |
|---|---|---:|
| **A. Shared encoder** | Retain as the control. Log per-task losses and shared-layer gradient norms; use the same training budget and checkpoint criterion across comparisons. | Control |
| **B. Shared + task adapter** | If objective balancing alone is insufficient, keep the encoder shared and add only a very small trajectory-specific bottleneck/normalization adapter after the shared representation. Match or tightly cap added parameters; do not widen the backbone. | 2 |
| **C. PCGrad** | Apply only if repeated saved-epoch diagnostics show negative shared-gradient cosine during training. Current sign differs between train and validation, so PCGrad is premature. | 3 |
| **D. Loss uncertainty weighting** | First objective ablation after logging raw and weighted task losses/gradient norms. It directly tests the measured scale imbalance without widening the network; constrain learned weights and report them per epoch. | **1** |

Before any new run, minimally repair instrumentation: log main intent BCE, proposal BCE, ambiguity regularizer, raw/weighted trajectory loss, and shared gradient norms every epoch; save epoch 1/5/10 snapshots; select/checkpoint on a predeclared trajectory-aware criterion. These are audit/protocol recommendations only, not executed.

## 6. Final conclusion

**Most likely: A. Loss imbalance**, with **B. gradient conflict / representation shift** as a likely compounding factor. Evidence: at saved epoch 14, `λ=1` leaves weighted trajectory SmoothL1 around `9e−5`–`2e−4`, while the intention objective is `1e−2` on the memorized train subset and `2.35` on a balanced validation subset; the trajectory-to-intention shared-gradient norm ratio is only `2.3e−3` on train and `7.4e−5` on validation. The held-out cosine is negative, but train cosine is positive, so a universal PCGrad diagnosis is not established.

**C. Architecture limitation** is plausible: the joint model routes both tasks through a changed fused trajectory input, and its test representation variance is much larger. **D. Training budget** is a secondary confound: joint training ran 15 epochs versus 20 for trajectory-only, although its selected epoch 14 was already its lowest validation ADE. Therefore “more epochs alone” is not supported as the main fix.

Bottom line: the ≈24 vs ≈11 px gap is best explained by the trajectory task receiving far too little shared-parameter optimization relative to intention, compounded by a shared representation that generalizes unstably. The saved artifacts do **not** establish epoch-by-epoch gradient conflict; that requires the missing epoch snapshots, not a new training run.

**Diagnostic invocation:**

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python scripts/audit_joint_model_gradients.py \
  --gradient-samples 512 --feature-samples 2048 --batch-size 256
```
