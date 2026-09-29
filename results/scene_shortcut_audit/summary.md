# Scene shortcut and crossing-context audit summary

## Scope and data firewall

All new analyses used only the clean train and validation archives. No official-test archive, prediction file, or metric was loaded. A0/J0 validation inference uses the existing frozen checkpoints; the only fit is the explicitly requested small scene-only diagnostic MLP.

## Answers to the audit questions

1. **What is the 512-D feature?** A full-frame, first-decodable-frame ResNet-18 ImageNet embedding, globally pooled after replacing `fc` with identity. See `scene_feature_provenance.md`.
2. **Single frame or temporal?** Single frame per video; not a temporal representation.
3. **Global or pedestrian-local?** Global frame; no pedestrian crop or mask is applied by the extractor.
4. **Within-video identity:** train 144/144 and validation 25/25 videos have exactly identical stored features across all rows. Same-video swap mean |Δp|=0.000000; ΔAUC=+0.000000.
5. **Scene-only validation:** mean AUC=0.8618 ± 0.0140; mean Brier=0.0790 ± 0.0068 across seeds 42/123/2024. On these same validation rows, frozen A0/J0 originals score AUC=0.8657 ± 0.0117, Brier=0.0776 ± 0.0095; the scene-only probe is close to the full model's validation AUC.
6. **Same-video swap:** ΔAUC=+0.000000 ± 0.000000; ΔBrier=+0.000000 ± 0.000000; mean |Δp|=0.000000.
7. **Cross-video swap:** ΔAUC=-0.5379 ± 0.0593; ΔBrier=+0.0756 ± 0.0170; mean |Δp|=0.1891.
8. **Video identity:** validation nearest-centroid top-1=1.0000, top-5=1.0000 among known validation video IDs. This is within-split identity retrieval, not unseen-video generalization.
9. **Video label prior:** validation has 25 videos and 40 pedestrian tracks; 2 videos are all-negative, 22 all-positive, and 1 mixed. Unweighted video positive-rate variance=0.0904 (range 0.000–1.000). Leave-one-pedestrian-out within-video prior AUC=0.8997, Brier=0.0806; this uses other validation pedestrians and is descriptive, not a deployable held-out-video prediction.
10. **Local versus far background:** status `not_run_raw_frames_unavailable`. No JAAD RGB clips were accessible in the workspace or mounted media locations. A 3.7-TB NTFS partition is detected as /dev/sda2 but is unmounted; it was not mounted or modified. No masks or R0/R1/R2 metrics are fabricated.
11. **Why did No Scene lose about 0.09 AUC?** The current scene input is necessarily a video-level global prior. The swap and scene-only results determine whether it is predictive/useful, but do not by themselves distinguish road semantics from background/domain identity. The local-vs-far test is unavailable unless source clips are mounted.
12. **Final current classification:** **background shortcut / video-level prior risk high; local crossing semantics remain unverified.** This describes evidence and uncertainty; it is not a claim that the model learned a novel scene method.

## Interpretation and next direction

The feature construction prevents same-video replacement from changing the input: every sample in a video receives exactly the same vector. Therefore a near-zero same-video swap is a construction invariant, not evidence that the model ignores scene context. Cross-video swap, scene-only performance, identity retrieval, and video label priors must be read together.

If cross-video replacement and video priors are strong, prioritize background-shortcut suppression and video/domain robustness before adding scene complexity. If local masking later shows the pedestrian vicinity carries predictive signal, preserve that local road/crosswalk evidence while regularizing invariance to distant background. Do not claim local crossing semantics from this audit until RGB interventions are completed.

## Machine-readable outputs

- `scene_feature_provenance.md`
- `scene_feature_similarity.json`
- `scene_only_probe.json`
- `same_video_scene_swap.json`
- `cross_video_scene_swap.json` (same combined file content, separated for convenient review)
- `video_identity_probe.json`
- `video_label_prior.json`
- `local_background_mask_audit.json`

## Limits

Validation videos are a finite set of 25 domains, and the scene feature is constant within each video. Scene-only and video-prior diagnostics are therefore not independent-row evidence. The local mask intervention remains unavailable because source clips are absent from accessible workspace paths; the detected 3.7-TB NTFS partition is unmounted and was not mounted by this audit.

The previously quoted AUC≈0.7644 was not assumed to be a validation metric. For this audit, the checkpoint outputs above reproduce the stored validation-selection values for the named A0/J0 checkpoints; do not compare the two numbers unless their split and selection protocol are confirmed to match.
