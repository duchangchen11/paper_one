# Clean J0 component dependency map

This map follows `JointTransformerSceneGate.forward` in
`src/models/joint_transformer_gate.py`; labels below describe actual tensor
flow, not module names.

```text
target_obs || target_abs_obs (8 features / observed step)
  -> target_projection + learned position embedding
  -> 3-layer target Transformer encoder
  -> last observed token = target_context ------------------------------┐
                                                                         │
scene_feat (512)                                                         │
  -> scene_encoder: Linear -> LayerNorm -> GELU -> Dropout               │
  -> scene_context --------------------------┐                           │
                                              │                           │
neighbor_obs (8 slots x 15 x 4)               │                           │
  -> shared neighbor GRU per slot             │                           │
  -> neighbor_hidden * neighbor_mask          │                           │
  -> masked mean, empty rows forced to zero    │                           │
  -> social_context --------------------┐     │                           │
                                        │     │                           │
target_context + scene_context + social_context                           │
  -> proposal_fusion (Linear -> GELU -> Dropout)                         │
  -> proposal_head -> prior_logit -> sigmoid -> Bernoulli entropy -------┤
                                        │                                │
target_context + scene_context + entropy                                 │
  -> uncertainty gate MLP -> sigmoid -> sample-dependent scalar g        │
                                        │                                │
target_context + scene_context + (g * social_context)                    │
  -> main fusion (Linear -> GELU -> Dropout)                              │
  -> intent_head -> main intent_logit                                     │
  -> trajectory head together with target_context -> future_pred         │
```

## Exact dependencies

- `prior_logit` is produced from the concatenation
  `[target_context, scene_context, social_context]` via `proposal_fusion` and
  `proposal_head`.
- The gate input is `[target_context, scene_context, entropy(prior_logit)]`.
  The gate does not directly receive `social_context`.
- The gate multiplies only `social_context`; it does not weight the scene or
  target features. The main fusion receives
  `[target_context, scene_context, gate * social_context]`.
- The final intention prediction is produced only by `fusion` and
  `intent_head`; `prior_logit` is not directly added to `intent_logit`.
- The proposal branch has two actual paths to the result: its auxiliary BCE
  loss during training, and its prior entropy into the uncertainty gate.
- Ambiguity regularization is
  `0.5 * (mean(prior_logit^2) + mean(intent_logit^2))` on the separate
  ambiguous training batch. It is not a separate prediction head or a
  validation/test term.
- In Clean J0, the objective is main intent BCE + `0.5 *` proposal BCE +
  `0.2 *` ambiguity regularizer + `0 *` trajectory loss. The trajectory head
  remains in the forward graph but has no trajectory supervision.

## Interventions frozen for this study

| Arm | Functional intervention | Paths deliberately retained |
|---|---|---|
| A1 No Scene | Replace `scene_context` with `zeros_like` before every downstream use | Scene encoder/module and all fusion widths |
| A2 No Social | Replace `social_context` with `zeros_like` before proposal and main fusion | Neighbor encoder/module and all fusion widths |
| A3 No Proposal Loss | Set proposal auxiliary BCE weight to zero | Proposal forward, prior entropy, gate, fusion |
| A4 No Adaptive Gate | Replace sample-dependent `g` by fixed `0.5` | Both scene/social contexts, proposal branch, gate/fusion modules |
| A5 No Ambiguity | Set ambiguity regularizer weight to zero | Model architecture and all ordinary training paths |

For A4, `0.5` is neutral for the implemented `g * social_context`
interpolation: it removes the learned sample-specific scale without setting
the social branch to either zero or full strength. This is intentionally an
adaptive-vs-neutral-gate comparison, not a gate-module deletion.
