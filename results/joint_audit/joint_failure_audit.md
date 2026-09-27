# Joint intention–trajectory failure audit

## Scope and outcome

Read-only audit of the saved seed-123 JAAD runs and source code. No model code was changed, no model was trained, and no optimizer step was run. The new gradient diagnostic uses `autograd.grad` on frozen checkpoint weights only.

On the matching saved experiment family, the trajectory-only Transformer obtains test ADE/FDE **11.0585/19.5931 px**, while the joint model obtains **25.0303/42.6616 px** (about 2.26× ADE and 2.18× FDE). The evidence most strongly supports **A. loss/gradient-scale imbalance**, compounded by a substantially changed joint feature flow and unstable feature scale. Broad training-set gradient conflict is not supported by the 10-batch mean; one previously measured balanced validation subset had negative cosine, so held-out conflict/generalization shift remains a secondary concern.

## 1. Architecture comparison

| Component | trajectory-only | joint |
|---|---|---|
| Input | 8-D target observation history (`target_obs` + `target_abs_obs`) and 512-D scene feature | Same target and scene inputs, plus neighbor histories/masks for social context; intention labels supply supervision |
| Encoder | Linear projection → 3-layer Transformer, hidden size 128, 4 heads; separate scene MLP, 512→128 | Same *shape* of target Transformer and scene MLP, plus neighbor GRU, proposal fusion/head, uncertainty gate, and task fusion |
| Hidden dimension | 128 | 128 |
| Trajectory decoder | MLP: `[target_context, scene_context]` → 15×2 normalized future coordinates | MLP: `[fused_context, target_context]` → 15×2; fused context combines target, scene, gated social context |
| Intention head | None | Main intent head on fused representation plus auxiliary proposal-intent head used by the uncertainty gate |
| Shared parameters | No intention task; trajectory loss updates its trajectory model | Target projection/Transformer, scene/social and fusion/gating pathway are exposed to both task objectives; separate final intent and trajectory heads |
| Task-specific parameters | Whole model is trajectory-only | Main intent head and trajectory head are task-specific; proposal/gate pathway also mediates the trajectory input |

**Answers.**

1. The joint branch does **not** reuse the trained trajectory-only encoder weights. It has a structurally matching 128-D, 3-layer target Transformer, but the standard joint checkpoint has no trajectory initialization checkpoint recorded; direct comparison found **0/36** matching Transformer tensors equal. It is a separately learned encoder, not a shared copy of the baseline encoder.
2. Yes. The trajectory feature flow changes from `[target_context, scene_context]` to `[fused(target, scene, gated social), target_context]`. The intent objective also updates shared upstream modules during joint training.
3. The final `intent_head` output itself is not an input to trajectory prediction, so there is no direct `intent_logit → future_pred` path. Indirect influence exists because the intent losses update shared target/scene/social/fusion/gate parameters; additionally, the proposal logit entropy affects the gate and therefore can change the trajectory input.

## 2. Loss and scale audit

The standard joint run uses `prior_weight=0.5`, `traj_weight (λ)=1.0`, and `ambiguous_weight=0.2` throughout. Training optimizes main BCE + weighted proposal BCE + λ·SmoothL1 + the ambiguity-logit regularizer on separate ambiguous samples. The saved epoch history has only a **combined validation loss**; it has no per-epoch intent/trajectory components or training loss components. The selected checkpoint is epoch 14; only the selected checkpoint was saved, so epoch 1/5/10 weights are unavailable. See [`loss_curve.json`](loss_curve.json), where absent components are explicitly `null`.

On the current saved checkpoint, the new 10 random trainer-matched training batches give mean component values:

| Quantity | Mean (across 10 batches) |
|---|---:|
| Main intent BCE | 0.00324 |
| Weighted proposal BCE (`0.5×`) | 0.00269 |
| Weighted ambiguity regularizer (`0.2×`) | 0.00156 |
| Combined intent objective | 0.00735 |
| Weighted trajectory SmoothL1 (`λ=1`) | 0.0000981 |
| Intent / trajectory scalar-loss ratio | ≈75× |

These sampled training losses are not the missing epoch curve and are not validation estimates. More decisively, their shared-parameter gradient norms differ by about **435×** in favor of intent (see next section). Thus `λ=1` is not balancing the two tasks at this checkpoint. Because coordinates are normalized and SmoothL1 is mean-reduced, the numeric λ alone is not meaningful; the measured weighted shared-gradient ratio is the stronger evidence. Earlier audit subsets also showed a train loss ratio ≈121× and a balanced validation ratio ≈14,117×, indicating a large train/validation confidence gap rather than a stable per-example loss scale.

The `run_epoch()` helper has a latent issue: a nonzero `ambiguous_weight` overwrites its supervised loss. The actual training loop does not use that helper for training, and validation calls it with the default zero, so this does not explain the saved run.

## 3. Gradient conflict analysis

[`analyze_joint_gradient_conflict.py`](../../scripts/analyze_joint_gradient_conflict.py) measures the configured intent objective and λ-weighted trajectory SmoothL1 over shared parameters on 10 independent 512-example class-balanced training batches. No optimizer was created and parameters were not updated. Results are in [`gradient_conflict.json`](gradient_conflict.json); standard deviations use sample `ddof=1`.

| Measure | Mean | SD | Min | Max |
|---|---:|---:|---:|---:|
| Cosine similarity `cos(g_intent,g_traj)` | **0.1277** | 0.1125 | −0.0846 | 0.3324 |
| `||g_intent||` | 0.10423 | 0.06097 | 0.05077 | 0.21361 |
| `||g_traj||`, λ-weighted | 0.0002362 | 0.0000367 | 0.0001873 | 0.0003161 |
| `||g_intent|| / ||g_traj||` | **454.39×** | 284.12× | 205.87× | 956.33× |

Nine of ten training batch cosines are positive and one is mildly negative. The mean indicates weak positive alignment, not pervasive train-batch conflict. However, the earlier fixed balanced validation subset had cosine **−0.322** and intent/trajectory gradient ratio ≈13,540×. Taken together, conflict is data/split-dependent and deserves validation monitoring, but the consistent and much stronger signal is gradient-scale imbalance. Per-epoch 1/5/10 cosines cannot be recovered without epoch snapshots.

## 4. Representation analysis

The new diagnostic samples 1,000 training examples uniformly without replacement and compares the saved encoders and actual decoder inputs. Population feature standard deviation is across all scalar feature entries; norm statistics are across samples.

| Representation | Width | Feature mean | Feature SD | Mean per-dimension variance | L2 norm mean ± SD |
|---|---:|---:|---:|---:|---:|
| Trajectory-only target encoder output | 128 | −0.0880 | 1.8212 | 0.07459 | 20.625 ± 0.397 |
| Joint target encoder output | 128 | −0.0712 | 2.2431 | **4.1716** | 24.504 ± 6.652 |
| Trajectory-only decoder input `[target, scene]` | 256 | 0.0728 | 1.3294 | 0.03901 | 21.299 ± 0.384 |
| Joint trajectory decoder input `[fused, target]` | 256 | 0.6834 | 2.3742 | **3.3313** | 38.351 ± 9.579 |

The joint target representation has roughly 56× the mean per-dimension variance and far greater sample-to-sample norm dispersion; its trajectory decoder input also has much larger scale/dispersion. This is **not representation collapse** (variance did not vanish). It is evidence of a major representation distribution/scale shift, not proof by itself that this shift caused the ADE gap. An earlier 2,048-test-sample diagnostic showed the same direction of change.

## 5. Reliability and trajectory error/loss

Existing motion-adjusted reliability results give adjusted-u Spearman correlation with pixel ADE/FDE of **0.229/0.199** on test (`n=18,331`), with video-cluster 95% CI for ADE correlation `[0.117, 0.312]`. On validation (`n=2,636`), correlations are **0.012/−0.005**, not significant. Test adjusted-u bins are not monotonic; raw uncertainty and observed motion correlate more strongly with ADE (0.428 and 0.450 respectively).

These scores come from a frozen three-seed **zero-scene trajectory-only ensemble**, not the joint checkpoint. Existing artifacts contain aggregate correlation/bin summaries but no aligned per-example adjusted-u plus joint ADE, and no per-sample training trajectory losses. Therefore `adjusted_u` vs joint ADE and `adjusted_u` vs training trajectory loss are **not computable from saved artifacts**. Reliability-aware weighting remains plausible but unproven for this joint task; establish a training-only/cross-fitted relation to per-sample joint loss before deciding the weighting direction. Details: [`reliability_task_relation.json`](reliability_task_relation.json).

## 6. Controlled ablation designs (proposals only)

- **Experiment A — freeze encoder, train trajectory head only.** Freeze the joint feature-producing path (target/scene/social/gate/fusion modules); optimize only `traj_head` with trajectory loss. This asks whether a fixed joint representation contains enough trajectory information for a decoder to recover accuracy. If it cannot, the issue is upstream representation/feature flow, not merely a poorly adapted output head.
- **Experiment B — freeze/disable intent-task updates.** Freeze intent-specific output/proposal parameters and train shared feature modules plus `traj_head` using trajectory loss only; do not backpropagate intent loss into shared parameters. This isolates whether continued intention gradients are suppressing trajectory learning. Keep the forward social/gate pathway fixed and document it, since it also supplies trajectory features.
- **Experiment C — freeze trajectory head during intention updates.** Freeze `traj_head`, train the intent objective through the shared trunk, and track held-out ADE from the fixed trajectory head. ADE degradation isolates drift in shared representations under intent updates; do not freeze the shared trunk in this test.

No ablation was run. Keep model size fixed and compare against a trajectory-only objective/lambda control.

## 7. Evidence-based diagnosis and next-method ranking

| Rank | Candidate | Evidence-based rationale |
|---:|---|---|
| 1 | **A. Loss balancing** (log components/norms first, then uncertainty weighting or dynamic λ) | Shared intent gradient is ≈435× trajectory gradient on sampled training batches despite `λ=1`; largest direct and repeatable imbalance signal. |
| 2 | **C. Architecture decoupling** (shared encoder + small task adapter / less altered trajectory path) | Joint trajectory input differs from baseline, encoder weights are independently learned, and joint representations have much larger scale/dispersion. Use only a parameter-matched controlled test. |
| 3 | **D. Reliability-aware trajectory weighting** `L_intent + λ(u)L_traj` | Adjusted-u has modest test association with trajectory error but no validation global association, is not joint-specific, and has no saved relation to training loss. First build a leakage-safe per-sample training relation. |
| 4 | **B. PCGrad / gradient surgery** | Training cosine averages positive; only one of ten sampled training batches is negative. A prior validation subset is negative, so monitor it, but current evidence does not justify making PCGrad the first intervention. |

**Conclusion:** Primary diagnosis is **A. loss imbalance**. **C. representation conflict/scale shift** and **D. architecture/feature-flow change** are plausible compounding mechanisms, not isolated causal findings. **B. gradient conflict** is not dominant across sampled training batches but may occur on held-out data. A simple budget increase is not supported as the first response: the joint model already reached its best validation ADE around epoch 14, and a prior λ=5 run remained near 24.7 px ADE.

## Verification

- Model/training source modifications: none.
- Large-scale training: not run.
- Diagnostic: `/home/lrj/anaconda3/envs/ped_intent/bin/python scripts/analyze_joint_gradient_conflict.py` (CUDA; optimizer not created; parameters not updated).
- Tests: **87 passed, 0 failed** (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .../python -m pytest tests -q`; 27 existing Transformer nested-tensor warnings). The plain command first failed before collection because the system ROS `launch_testing` pytest plugin imports missing `lark`; disabling external plugin autoload allowed the project test suite to run.
