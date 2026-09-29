# Scene feature provenance audit

## Verified extraction path

The checked-in implementation is `scripts/extract_scene_features.py`. It requires `--video-root`; for each unique `scene_id` it opens `<video-root>/<scene_id>.mp4` with OpenCV and calls `read()` once on a new capture. Therefore the source is the first decodable RGB frame, not a per-sample observation frame.

The exact historical absolute `--video-root` argument was not saved in the repository; the code-level path convention is verified, while that machine-specific root is not recoverable from the current manifest/logs.

## Image and feature construction

- Image source: full-frame JAAD clip `<scene_id>.mp4`; the extractor does not crop or mask the pedestrian.
- Color: OpenCV BGR is converted to RGB.
- Backbone: torchvision ResNet-18, `weights=ResNet18_Weights.DEFAULT` (IMAGENET1K_V1); pretrained ImageNet weights, URL `https://download.pytorch.org/models/resnet18-f37072fd.pth`.
- Extraction layer: replace `fc` with `Identity`; ResNet global average pooled output is 512-D.
- Current audit runtime: PyTorch 2.5.1+cu124, torchvision 0.20.1+cu124; transform resize=[256], center crop=[224], interpolation=InterpolationMode.BILINEAR, ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
- Backbone is put in eval mode and is not fine-tuned during extraction.
- Extract once per unique video ID; `feature_by_scene[scene]` is copied into every row for that video when the split NPZ is written.
- Saved representation is float16 in the processed NPZ and loaded as float32 by the dataset class.

## Per-video consistency on allowed splits

| Split | Rows | Videos | Exact-constant videos | Zero-feature videos |
|---|---:|---:|---:|---:|
| train | 24556 | 144 | 144 | 0 |
| validation | 2636 | 25 | 25 | 0 |

## Reproducibility limits

The extractor does not save a run manifest with the historical torchvision version, exact resolved weights checksum, input clip root, source frame hashes, or preprocessing configuration. The current code resolves `DEFAULT` to the weight enum shown above in this audit runtime; that is code-level provenance, not proof of the exact package/weight file used during the original extraction.

Processed archives inspected: `data/processed/jaad_sequences_scene_15x15/train.npz` and `val.npz` only. The official test archive was not opened.
