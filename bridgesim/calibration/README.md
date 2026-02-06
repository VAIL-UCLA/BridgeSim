# Calibration Workflow TODO

- [x] Parse OpenScene/RAP logs to extract paired 3D rasterization frames and corresponding real-world camera images.
- [x] Output aligned image-pair metadata for downstream calibration experiments.
- [ ] Add flow-matching based feature alignment to refine pair correspondences between synthetic and real frames.
- [ ] Introduce a script to build triplets linking 3D rasterization images, MetaDrive simulation renders, and real-world images for joint evaluation.

## Flow-Matching Domain Adaptation Plan

- **data**: dataset utilities to read rasterized/real pairs from the JSONL manifests produced by `build_openscene_rap_image_dataset.py` and `build_diverse_sub_dataset.py`, plus image transforms and a collate function that keeps camera metadata.
- **encoders**: wrappers around pretrained image encoders (e.g., the DINO backbone in `RAP/navsim/agents/rap_dino/bevformer/image_encoder.py`) and an optional FPN neck to expose multi-scale features for flow matching.
- **flow_model**: a conditional vector-field network that learns a flow from rasterized features (source) to real-image features (target) using flow-matching objectives.
- **training**: trainer loop, loss functions, optimizer/scheduler setup, checkpointing, and logging hooks so the model can be trained end-to-end or with frozen encoders.
- **evaluation/inference**: scripts to run the trained flow on held-out pairs and dump aligned features or intermediate visualizations for inspection.
- **configs/scripts**: small CLI entry points to launch training/eval with YAML/JSON configs and sensible defaults under `calibration/flowmatch`.
