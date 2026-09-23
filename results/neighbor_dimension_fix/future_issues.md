# Future issues (record only; not changed in this controlled experiment)

## Issue A: neighbor visibility is not used by the neighbor GRU

`neighbor_visible_mask` is returned by the social models but does not mask or otherwise affect neighbor GRU encoding. During dataset construction, missing neighbor observations are filled using the first visible position, which can create unrealistic temporal sequences. This should be investigated separately from the neighbor/time axis correction.

## Issue B: uncertainty entropy depends on social context

The current proposal entropy is computed from a proposal that already includes `social_context`, and that entropy then determines the weight applied to `social_context`. This creates a feedback-like dependency in the gating design. A later controlled study could estimate uncertainty from target and scene features first, then use it to gate a social residual.

## Issue C: scene feature is static at video level

The current scene feature is a frozen ResNet-18 embedding extracted from the first frame of each video and shared by every sample from that video. It is video-level static scene context rather than sample-specific visual context.
