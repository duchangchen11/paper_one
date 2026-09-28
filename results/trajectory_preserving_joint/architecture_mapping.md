# Trajectory-only architecture mapping

Reference source: `src/models/trajectory_transformer.py`, class
`SceneTrajectoryTransformer`; training implementation:
`scripts/train_trajectory_transformer.py`.

## Forward data path

1. Inputs are `target_obs` with shape `[B, 15, 4]`, `target_abs_obs` with shape
   `[B, 15, 4]`, and `scene_feat` with shape `[B, 512]`. The trainer concatenates
   the two target histories along their feature axis to form `[B, 15, 8]`.
2. `input_projection`: linear projection `8 → 128`; add the learned positional
   embedding `[1, max_obs_len, 128]`.
3. `temporal_encoder`: three-layer, four-head Transformer encoder; take the last
   time-step output as `target_context` `[B, 128]`.
4. `scene_encoder`: `Linear(512,128) → LayerNorm(128) → GELU → Dropout(0.1)`;
   output `scene_context` `[B,128]`.
5. The original decoder input is exactly
   `traj_input = concat(target_context, scene_context)` `[B,256]`.
6. `decoder`: `Linear(256,128) → GELU → Dropout(0.1) → Linear(128,30)`;
   reshape to normalized future coordinates `[B,15,2]`.

Thus the reference forward is exactly:

```text
target_context = temporal_encoder(input_projection(target_history) + position_embedding)[:, -1]
scene_context = scene_encoder(scene_feat)
traj_input = concat(target_context, scene_context)
future_pred = decoder(traj_input).reshape(B, 15, 2)
```

## Preservation design

The new model will contain the original `SceneTrajectoryTransformer` as a
backbone and evaluate these same modules in the same order. Its original decoder
will receive only `[target_context, scene_context]`. Neither social features,
intent logits, nor an intention-fused representation will enter the trajectory
path. The frozen backbone will remain in evaluation mode (dropout disabled) even
while the new intention head is trained.

P1 uses only `target_context` for intention. P2 uses
`concat(target_context, scene_context)`. The new intention branch is the only
trainable part. An initialization equivalence test over at least 1,024
validation samples is required before either intention run may begin; its result
is recorded separately in `equivalence_test.json`.

## Per-seed pretrained checkpoints

Checkpoint paths, SHA256 values, selected checkpoint validation metrics, and
existing standalone test ADE/FDE values are recorded in
`trajectory_reference.json`. Each P1/P2 seed must load only the checkpoint with
the same seed.
