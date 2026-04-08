# Q&A

## General

**Q: What is BridgeSim?**  
A: BridgeSim is a cross-simulator closed-loop evaluation platform for end-to-end autonomous driving policies. It lets you evaluate models trained on NavSim or Bench2Drive across multiple real-world datasets (NavSim, Waymo, nuScenes, and more) in a unified interface built on MetaDrive.

**Q: What is the OL-CL gap?**  
A: The open-loop / closed-loop gap refers to the discrepancy between how a model performs when evaluated on pre-recorded data (open-loop) versus when it actually controls the vehicle and its actions affect the scene (closed-loop). BridgeSim is designed to measure and bridge this gap.

**Q: Which datasets are supported?**  
A: NavSim / OpenScene, Bench2Drive, nuScenes, and Waymo.

---

## Installation

**Q: Which conda environment should I use?**  
A: It depends on the model:
- `mdsn` — DiffusionDrive, DiffusionDriveV2, LTF, TransFuser, DrivoR, LEAD
- `b2d` — UniAD, VAD
- `rap` — RAP

**Q: I get `GLIBCXX_3.4.xx not found` on a headless server. How do I fix it?**  
A:
```bash
mkdir -p /usr/lib/dri
ln -s /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so /usr/lib/dri/swrast_dri.so
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 $(conda info --base)/envs/mdsn/lib/libstdc++.so.6
```
Replace `mdsn` with your env name (`b2d` or `rap`) as appropriate.

**Q: Panda3D can't find pedestrian or traffic cone models. How do I fix it?**  
A: Register the model paths manually (replace `<ENV>` and the metadrive path):
```bash
echo "model-path /path/to/metadrive/metadrive/assets/models/pedestrian" >> /opt/conda/envs/<ENV>/lib/python3.9/site-packages/panda3d/etc/Config.prc
echo "model-path /path/to/metadrive/metadrive/assets/models/traffic_cone" >> /opt/conda/envs/<ENV>/lib/python3.9/site-packages/panda3d/etc/Config.prc
```

**Q: How do I download the model checkpoints?**  
A:
```bash
huggingface-cli download sethzhao506ucla/BridgeSim --local-dir ckpts/BridgeSim
```

---

## Data Preparation

**Q: What scene filters are available for NavSim/OpenScene conversion?**  
A: Two filters are provided under `converters/openscene/filter/`:
- `navhard_two_stage.yaml` — NavHard split (challenging scenarios)
- `navtest.yaml` — NavTest split

**Q: What does `--interpolate` do in the OpenScene converter?**  
A: It upsamples the scenario from 2Hz to 10Hz, which is required for smooth closed-loop evaluation in MetaDrive.

**Q: How many future frames should I extract?**  
A: Use `--num-future-frames-extract 220` for full scenarios. For quick testing, `40` is sufficient.

---

## Evaluation

**Q: What traffic modes are available?**  
A:
- `no_traffic` — no other vehicles
- `log_replay` — replay the recorded traffic from the log
- `IDM` — simulate traffic using the Intelligent Driver Model

**Q: What is `--replan-rate`?**  
A: The number of simulation steps between model inference calls. A rate of 1 means the model is queried every step; higher values reduce compute but may degrade performance.

**Q: What is `--ego-replay-frames`?**  
A: The number of frames at the start of a scenario where the ego vehicle follows the logged trajectory before the model takes over. Useful for warm-starting the scene.

**Q: Can I run open-loop evaluation?**  
A: Yes, pass `--eval-mode open_loop` to `unified_evaluator.py`.

**Q: How do I resume a batch evaluation that was interrupted?**  
A:
```bash
bash scripts/evaluator/run_batch_eval.sh [model_type] [scenario_root] --resume
```

**Q: I get OOM errors. What should I do?**  
A: Use `CUDA_VISIBLE_DEVICES` to select a specific GPU and avoid running multiple evaluations on the same GPU simultaneously:
```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/evaluator/run_eval.sh transfuser /path/to/scenario
```

**Q: The scenario path is wrong. What's the correct format?**  
A: Point to the individual scenario directory, not its parent:
```
✓ /path/to/scenarios/Accident_Town03_Route101_Weather23
✗ /path/to/scenarios
```

---

## Models

**Q: Does RAP require a HuggingFace login?**  
A: Yes. RAP uses `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`, which requires access approval and a login token:
```bash
huggingface-cli login --token hf_xxxxxxxx
```

**Q: What `--image-source` options does RAP support?**  
A: `metadrive` (rendered by the simulator) and `rasterized_3d`.

**Q: What `--planner-type` options does TCP support?**  
A: `learned` and `reactive`.

**Q: Can I add a new model?**  
A: Yes. Create an adapter in `bridgesim/evaluation/models/` inheriting from `BaseModelAdapter`, implement `load_model()` and `run_inference()`, then register it in `unified_evaluator.py`. See `bridgesim/evaluation/README.md` for details.
