# Local Crossing Context vs Far Background Audit

## Scope
Only train and validation were opened. Official test was not loaded or used for any model/scale selection. ResNet-18 is frozen. R0 is the actual obs_end_frame full image; R1 retains the centered 4x pedestrian bbox neighborhood; R2 is the complementary far-background image from that same frame. The 2x/6x variants are validation-only sensitivity checks.

## Data and indexing
- Raw videos: /media/lrj/54926A1D926A0438/ped_intent_project/data/raw/JAAD/JAAD_clips (346 clips).
- scene_id maps directly to the same filename stem, normally scene_id.mp4.
- obs_end_frame is the original XML frame@number, unchanged by preprocessing; exact train/val target-track matches and bounds are in frame_index_mapping.md.
- Features use ImageNet ResNet-18, fc=Identity, 512-D, eval mode, and the same resize/crop/normalization as the original extractor.

## Scene-only probes (train to validation)
| Representation | AUC mean ± SD | Brier mean ± SD |
|---|---:|---:|
| Old static first frame | 0.8618 ± 0.0140 | 0.0790 ± 0.0068 |
| Current full frame | 0.7798 ± 0.0125 | 0.1354 ± 0.0076 |
| Local 4x | 0.5742 ± 0.0047 | 0.1203 ± 0.0017 |
| Far background 4x | 0.8400 ± 0.0087 | 0.1010 ± 0.0044 |

## Matched B0/B1/B2 training

| Arm | Validation AUC mean ± SD | Brier mean ± SD | Cross-video ΔAUC | Cross-video ΔBrier | Mean |Δp| |
|---|---:|---:|---:|---:|---:|
| B0 (R0_current_full) | 0.7120 ± 0.0407 | 0.1386 ± 0.0114 | -0.0336 | -0.0123 | 0.0884 |
| B1 (R1_local_4x) | 0.5875 ± 0.0313 | 0.1235 ± 0.0012 | +0.1311 | +0.0184 | 0.0754 |
| B2 (R2_background_4x) | 0.8441 ± 0.0134 | 0.0995 ± 0.0039 | -0.2990 | +0.0463 | 0.1274 |

All arms share the same seed-specific initialization, sampler sequence, 15 epochs, AdamW (lr 1e-3, weight decay 1e-4), gradient clip 5, and full Clean J0 architecture. Objective weights for trajectory, prior, and ambiguity are zero. Checkpoint selection uses raw validation AUC, with lower Brier within a 1e-4 tie. ADE/FDE are not selection criteria.

## Temporal variation and video identity
- Old static feature: same-video cosine is 1.000 and distinct-window exact duplicate fraction is 1.000.
- R0 current full frame: distinct observation-frame within-video cosine is 0.9367; exact duplicate fraction is 0.000, so it varies over time.
- Pairwise video-identity cosine AUC (higher means same-vs-different video identity is more separable): old_static=1.0000, R0_current_full=0.9881, R1_local_4x=0.8723, R2_background_4x=0.9803.
- Among newly extracted views, local 4x carries the least video identity; far background remains highly video-identifiable. Detailed same-pedestrian and different-pedestrian cosine values are in dynamic_scene_feature_similarity.json.

## 2x/4x/6x sensitivity (existing J0, intervention only)
| Bbox scale | Local AUC | Local Brier | Far-background AUC | Far-background Brier |
|---:|---:|---:|---:|---:|
| 2x | 0.5678 | 0.7432 | 0.7897 | 0.1307 |
| 4x | 0.6562 | 0.5544 | 0.7661 | 0.1633 |
| 6x | 0.6863 | 0.4090 | 0.7765 | 0.1740 |

These are validation-only replacements into J0 checkpoints trained on old static features, so they are distribution-intervention diagnostics, not fair newly trained model comparisons. The predeclared primary scale remains 4x. Probability shifts/correlations are retained in local_scale_sensitivity.json.

## Interpretation
- Scene-only AUC: old static 0.8618; current full 0.7798; local 4x 0.5742; far background 4x 0.8400.
- Matched training AUC: B0 full 0.7120; B1 local 0.5875; B2 background 0.8441.
- Evidence is classified as **background**. The strongest evidence is B2 outperforming B0 by +0.1320 while B1 trails B0 by -0.1245; B2 also loses 0.2990 AUC on mean under cross-video scene swap.
- Therefore the earlier approximately +0.09 scene gain is more consistent with a video/background prior than local crossing evidence. This does not mean the background is semantically irrelevant; it means its predictive association is highly video-dependent and brittle under cross-video replacement.
- Recommended next direction: **Prioritize background-shortcut suppression and scene-invariant intent learning; revisit local dynamic context after this confound is controlled.**

## Direct answers to the audit questions

1. Original video root: /media/lrj/54926A1D926A0438/ped_intent_project/data/raw/JAAD/JAAD_clips.
2. Mapping: scene_id is unchanged as the video stem, normally scene_id.mp4; 346 source clips were found.
3. obs_end_frame is copied unchanged from the original JAAD XML box frame integer; it is zero-based and indexes the source video, not the processed row array. Train and validation exact (scene_id, target_id, obs_end_frame) XML matches: 27,192/27,192. Missing target-track/frame matches: 0. Validation out-of-bounds frame indices: 0. Official test was not loaded.
4. Yes. R0 same-video cosine across distinct observation frames is 0.9367; exact duplicate fraction is 0.000.
5. Old static scene-only AUC: 0.8618.
6. Current full-frame scene-only AUC: 0.7798.
7. Local 4x scene-only AUC: 0.5742.
8. Far-background 4x scene-only AUC: 0.8400.
9. Matched validation AUC: B0 0.7120, B1 0.5875, B2 0.8441.
10. Matched validation Brier: B0 0.1386, B1 0.1235, B2 0.0995.
11. J0 intervention AUC by local scale 2x/4x/6x: 0.5678/0.6562/0.6863; far-background: 0.7897/0.7661/0.7765. Local varies with scale; background AUC is comparatively stable, but these J0 replacements are out-of-distribution.
12. Pairwise video-identity cosine AUC: old static 1.0000; current full 0.9881; local 0.8723; far background 0.9803.
13. Cross-video scene swap mean ΔAUC: B0 -0.0336, B1 +0.1311, B2 -0.2990; mean |Δp| is 0.0884/0.0754/0.1274.
14. The previous scene gain is most consistent with a video/background prior, not local-only evidence. Recommended next direction: Prioritize background-shortcut suppression and scene-invariant intent learning; revisit local dynamic context after this confound is controlled.

## Limitations
Scene-only rows are clustered within videos, so row-level SD is not a video-level confidence interval. J0 replacement results are out-of-distribution interventions. The matched arms control seed, initialization, sampling and training protocol, but checkpoint selection uses validation. Large feature arrays remain ignored local artifacts; committed manifests contain hashes and provenance.

## Artifacts
raw_video_location.json; raw_video_mapping_audit.json; frame_index_mapping.md; dynamic_scene_feature_similarity.json; local_background_feature_manifest.json; scene_probe_*.json; a0_current_scene_intervention.json; local_scale_sensitivity.json; initialization_match.json; sampler_match.json; matched_scene_representation_results.json; cross_video_robustness.json.
