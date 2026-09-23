# Scene-feature definition audit

## Verified source and construction

The current `scene_feat` is a **video-level static scene context** (equivalently, a **first-frame video-level scene embedding**):

1. `scripts/extract_scene_features.py` groups samples by `scene_id`.
2. For each JAAD video, it opens `<scene_id>.mp4` and reads one frame before any per-sample expansion; this is the video's first decodable frame.
3. The frame is RGB-converted and transformed with the default pretrained ImageNet `ResNet18_Weights` transform.
4. A torchvision ResNet-18 with pretrained ImageNet weights and `fc = Identity` produces a 512-D embedding.
5. That single embedding is looked up by `scene_id` and copied to every sample from that video.

It is **not** a dynamic-scene, frame-level, or sample-specific visual feature. This audit did not regenerate or alter scene features, processed NPZ files, or the ResNet backbone.

## Processed-data consistency check

Read-only checks on the existing clean 15×15 split files found:

| Split | Samples | Unique videos/scenes | Feature dimension | Max within-video embedding deviation | Finite values |
|---|---:|---:|---:|---:|---|
| Train | 24,556 | 144 | 512 | 0 | Yes |
| Validation | 2,636 | 25 | 512 | 0 | Yes |
| Test | 18,331 | 97 | 512 | 0 | Yes |

Thus, within every split, all samples assigned to the same `scene_id` carry exactly the same stored embedding. No processed data was modified.
