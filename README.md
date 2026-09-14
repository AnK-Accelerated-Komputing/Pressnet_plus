# Pressnet_plus

Staged multi-model learned surrogate for **press forming** structural simulation.

Pressnet++ is the code release accompanying the paper:

> **PRESSNET: A Forming Dataset for Structural Simulation in Pressed Blanks with Deep Learning Benchmarks** — Panta et al., *ASME IDEC/CIE 2025*, Anaheim, CA (`IDETC2025-163821`).

## What it does

Numerical simulation is integral to engineering design, replacing costly physical
prototyping with virtual iterations; yet high-fidelity methods such as the finite
element method (FEM) are computationally expensive, limiting how often they can run
inside a design loop. Press-forming is an especially demanding case, because a single
solve couples large visco-elastic deformation, moving-boundary contact, stress
relaxation, and thermal diffusion in one transient, multi-body analysis.

Learned surrogates promise a faster alternative, but existing forming surrogates
target a single field, a quasi-static setting, or a thermal-only response; none
reproduces the complete multi-stage, multi-physics forming trajectory. PressNet++
closes this gap using the PressNet dataset of 150 transient pressed-forming
trajectories (see the paper citation below for the canonical reference).

### Contributions

- We propose PressNet++, a **staged multi-model surrogate architecture** that
  partitions the forming trajectory into specialized networks for the pressing,
  dwell, and release regimes, together with a parallel network for thermal
  diffusion, and show that it outperforms a monolithic single-model baseline.
- We provide a **controlled benchmark of five architectures** spanning two
  families — graph neural networks and a transformer-based PDE solver — on the
  same transient, multi-physics, large-deformation task.
- We evaluate **generalization and stability** through in-distribution,
  parametric-extrapolation, and unseen-shape splits, and through a one-step
  versus rolled-out comparison that isolates error accumulation. Global-attention
  models are more accurate on geometries seen during training, whereas local
  mesh-based message passing generalizes better to unseen geometries.
- We report **architecture-specific hyperparameter ablations** and a
  **mesh-resolution study** that includes cross-resolution transfer between
  coarse and fine meshes.

### Architecture

PressNet++ is a **staged multi-model** architecture in which separate networks
handle the pressing, dwell, and release regimes, and a parallel network handles
thermal diffusion. We validated this design across five architectures spanning two
families — graph neural networks (GCN, Reg-DGCNN, MeshGraphNet, Dilated-DGCNN) and
the transformer-based solver Transolver.

Key findings:

- The staged formulation outperforms a monolithic baseline for every architecture.
  For Transolver, rolled-out y-displacement nRMSE drops from 6.57% to 3.20% in the
  coarse mesh and to 1.76% in the fine mesh.
- Architectures behave very differently out of distribution: Transolver is most
  accurate on die shapes seen in training, yet on unseen shapes MeshGraphNet is far
  more robust (9.47% versus 54.76% nRMSE).
- One-step accuracy below 0.4% displacement nRMSE for every model shows long-horizon
  stability issues and potential improvement through stabilization of autoregressive
  inference.
- Stress prediction has a much higher error than displacement in all scenarios.

## Results

### Training loss

![Training loss](training.png)

### Inference rollout

![Inference rollout](inference.png)

## Quick start

```bash
conda env create -f environment.yml
conda activate graph_env
```

Each architecture is trained **three times** — once per stage (Press / Dwell /
Release) — then stitched together at inference. Replace `--model` with the
architecture you want.

### Training (one command per stage per model)

```bash
# --- Graph neural networks ---
python -m pressnetpp.train --config configs/train_dilated_dgcnn.json --stage 1 --model dilated_dgcnn
python -m pressnetpp.train --config configs/train_transolver.json  --stage 1 --model transolver
python -m pressnetpp.train --config configs/train_gcn.json          --stage 1 --model gcn

# Stage 2 (Dwell) and Stage 3 (Release) use the same commands with --stage 2 / --stage 3
```

### Inference (3-stage rollout)

```bash
python -m pressnetpp.inference --config configs/inference_multi.json
```

Set `checkpoint_path_1/2/3` in the config to the three trained stage
checkpoints (Press / Dwell / Release). `core_model` can be a single string
(same architecture all stages), a list of three, or a dict with
`stage_1` / `stage_2` / `stage_3` keys.

## Verified

The package was smoke-tested end-to-end in `graph_env`:

- `import pressnetpp.train, pressnetpp.inference` ✓
- `TrajectoryDataset` loads the real coarse dataset (13,694 samples, 887 nodes,
  all expected fields) ✓
- One real training step (forward + backward + optimizer step) on real data
  passes for **encode_process_decode, gcn, transolver, regDGCNN_seg, and
  regpointnet_seg** ✓
- `dilated_dgcnn` requires the compiled CUDA `furthest_point_sampling`
  extension and needs a GPU (the extension's CPU fallback is not implemented);
  it runs on GPU in the same code path.

## Repository layout

```
Pressnet++/
  README.md
  LICENSE
  CITATION.bib
  environment.yml
  configs/                       # example JSON configs
  pressnetpp/
    train.py                     # train one stage (Press / Dwell / Release)
    inference.py                 # run the full 3-stage rollout
    models/                      # 5 benchmarked architectures + wrapper
    utilities/                   # dataset loader, eval, plot, anim, paraview
```

## Quick start

```bash
conda env create -f environment.yml
conda activate graph_env

# 1. Train stage 1 (Press) with Dilated-DGCNN
python -m pressnetpp.train --config configs/train_dilated_dgcnn.json \
    --stage 1 --model dilated_dgcnn

# 2. Repeat for stage 2 (Dwell) and stage 3 (Release), optionally changing --model

# 3. Run the full 3-stage rollout
python -m pressnetpp.inference --config configs/inference_multi.json
```

See `configs/*.json` for full hyperparameter sets; paths are user-configurable.

## Data

The PressNet dataset (150 trajectories × 15 die shapes × 10 geometric variations,
coarse/medium/fine meshes, 1500 time steps) is not redistributed here. The dataset
and its download instructions are part of the parent PressNet repository (see the
ASME paper citation below for the canonical reference). The thermal sub-dataset
(`datasets/data/thermal/s_quarter_1500_withthermal.h5`) lives in the parent repo.

> **Note on thermal:** the parallel thermal-diffusion network referenced in the
> paper is not included in this release. The thermal dataset is present, but the
> thermal model and its evaluation utilities live in a separate codebase.

## Citation

If you use this code or dataset, please cite:

```
@inproceedings{PRESSNET2025,
  author    = {Prince Panta and Saroj Belbase and Rachit Rijal and Bipin Shrestha and Nirmal Prasad Panta and Dikshya Parajuli and Rujal Acharya and Saugat Kafley and Amit Regmi and Ken Igeta and Akio Tanaka and Christopher McComb},
  title     = {{PRESSNET: A Forming Dataset for Structural Simulation in Pressed Blanks with Deep Learning Benchmarks}},
  booktitle = {Proceedings of the ASME 2025 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference (IDETC/CIE 2025)},
  year      = {2025},
  address   = {Anaheim, CA, USA},
  paperid   = {IDETC2025-163821},
  publisher = {American Society of Mechanical Engineers (ASME)},
  doi       = {10.1115/DETC2025-163821},
  url       = {https://doi.org/10.1115/DETC2025-163821}
}
```

## License

Copyright Accelerated Computing Pvt. Ltd. See `LICENSE`.
