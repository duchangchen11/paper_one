# Frame-index mapping audit

The JAAD preprocessor parses each XML box@frame integer as the original frame key, builds observation windows ending at obs_end, and appends obs_end unchanged as obs_end_frame. This is a source-video frame number, not a processed-array row index. OpenCV CAP_PROP_POS_FRAMES uses zero-based frame positions.

Train and validation exact (scene_id, target_id, obs_end_frame) XML matches: 27,192/27,192. Missing target-track/frame matches: 0. Validation out-of-bounds frame indices: 0. Official test was not loaded.
