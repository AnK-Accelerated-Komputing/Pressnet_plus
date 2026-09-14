"""Pressnet++ inference entry point.

Mirrors surrogateAI/inference3_multi.py. Runs the full 3-stage (Press /
Dwell / Release) rollout: three separately-trained checkpoints, one per
stage, are stitched together in continuous mode. Each stage may use a
different architecture.

Usage:
    python -m pressnetpp.inference --config configs/inference_multi.json
"""

import os
import torch
import pickle
import time
import json
import argparse
import gc
from torch.utils.data import DataLoader

from pressnetpp.models import press_model
from pressnetpp.utilities import press_eval, common
from pressnetpp.utilities.dataset import TrajectoryDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARAMETERS = {
    "press": dict(
        noise=0.003,
        gamma=1.0,
        field="world_pos",
        history=False,
        size=4,
        batch=1,
        model=press_model,
        evaluator=press_eval,
        loss_type="deform",
        slice=64,
        message_passing_steps=20,
        stochastic_message_passing_used="False",
        k_neighbor=20,
        dilated_k_sample=200,
    )
}


def _resolve_core_models(inference_config):
    core_cfg = inference_config.get("core_model", "encode_process_decode")
    if isinstance(core_cfg, str):
        return [core_cfg, core_cfg, core_cfg]
    if isinstance(core_cfg, (list, tuple)):
        if len(core_cfg) != 3:
            raise ValueError(f"core_model list must have exactly 3 entries, got {len(core_cfg)}")
        return list(core_cfg)
    if isinstance(core_cfg, dict):
        try:
            return [core_cfg["stage_1"], core_cfg["stage_2"], core_cfg["stage_3"]]
        except KeyError as e:
            raise ValueError(f"core_model dict missing key {e}; expected stage_1/stage_2/stage_3")
    raise TypeError(f"Unsupported core_model type: {type(core_cfg).__name__}")


def load_model(inference_config, params, checkpoint_path_1, checkpoint_path_2, checkpoint_path_3):
    core_names = _resolve_core_models(inference_config)
    core_name_1, core_name_2, core_name_3 = core_names
    print(f"[INIT] Stage 1 architecture: {core_name_1}")
    print(f"[INIT] Stage 2 architecture: {core_name_2}")
    print(f"[INIT] Stage 3 architecture: {core_name_3}")

    model1 = params["model"].Model(params, core_model_name=core_name_1)
    model2 = params["model"].Model(params, core_model_name=core_name_2)
    model3 = params["model"].Model(params, core_model_name=core_name_3)

    def safe_load(model, target_path, stage_name):
        possible_prefixes = []
        if os.path.isdir(target_path):
            possible_prefixes.append(os.path.join(target_path, "best_model_checkpoint"))
            possible_prefixes.append(os.path.join(target_path, "epoch_model_checkpoint"))
            possible_prefixes.append(os.path.join(target_path, "model_checkpoint"))
        else:
            possible_prefixes.append(target_path.replace("_learned_model.pth", ""))

        for prefix in possible_prefixes:
            test_file = prefix + "_learned_model.pth"
            if os.path.exists(test_file):
                print(f"[{stage_name}] Found checkpoint at {test_file}. Loading...")
                try:
                    model.load_model(prefix)
                    print(f"[{stage_name}] Success: Loaded model.")
                    return True
                except Exception as e:
                    print(f"[{stage_name}] Warning: Error loading from {prefix} ({e}). Trying next fallback...")
        print(f"[{stage_name}] CRITICAL ERROR: No valid checkpoint found in {target_path}")
        return False

    print("\n" + "=" * 60)
    print("--- Loading Models ---")
    print("=" * 60)
    success1 = safe_load(model1, checkpoint_path_1, "Stage 1 (Press)")
    success2 = safe_load(model2, checkpoint_path_2, "Stage 2 (Dwell)")
    success3 = safe_load(model3, checkpoint_path_3, "Stage 3 (Release)")

    if not (success1 and success2 and success3):
        raise RuntimeError(f"Failed to load all 3 models. Check checkpoint paths:\n"
                           f"  Stage 1: {checkpoint_path_1}\n"
                           f"  Stage 2: {checkpoint_path_2}\n"
                           f"  Stage 3: {checkpoint_path_3}")
    print("=" * 60 + "\n")

    def apply_patch(wrapper_model):
        original_forward = wrapper_model.learned_model.forward

        def patched_forward(*args, **kwargs):
            kwargs.pop("is_training", None)
            return original_forward(*args, **kwargs)

        wrapper_model.learned_model.forward = patched_forward

    for stage_model, stage_core in zip([model1, model2, model3], core_names):
        if stage_core == "transolver":
            apply_patch(stage_model)

    model1.to(device).eval()
    model2.to(device).eval()
    model3.to(device).eval()
    return model1, model2, model3


def squeeze_data(data):
    return {k: v.squeeze(0).to(device) for k, v in data.items()}


def validate_stage_alignment(loader1, loader2, loader3, num_samples=3):
    print("\n[VALIDATION] Checking stage data alignment...")
    for i, (batch1, batch2, batch3) in enumerate(zip(loader1, loader2, loader3)):
        if i >= num_samples:
            break
        cells_shape_1 = batch1["cells"].shape
        cells_shape_2 = batch2["cells"].shape
        cells_shape_3 = batch3["cells"].shape
        if not (cells_shape_1 == cells_shape_2 == cells_shape_3):
            raise ValueError(
                f"Stage misalignment detected at batch {i}!\n"
                f"  Stage 1 cells shape: {cells_shape_1}\n"
                f"  Stage 2 cells shape: {cells_shape_2}\n"
                f"  Stage 3 cells shape: {cells_shape_3}\n"
                f"Possible causes:\n"
                f"  1. Cache files (*_val_1.pt, *_val_2.pt, *_val_3.pt) are out of sync\n"
                f"  2. Different dataset versions were used\n"
                f"  3. Data processing was interrupted"
            )
        print(f"  Batch {i}: Aligned (shape: {cells_shape_1})")
    print("Stage alignment validation PASSED\n")


def validate_stage_continuity(pred_traj1, pred_traj2, pred_traj3,
                             traj2_input_stitched, traj3_input_stitched):
    warnings = []
    stage1_end_pos = pred_traj1["pred_pos"][-1]
    stage2_start_pos = traj2_input_stitched["curr_pos"][0]
    pos_gap_1_2 = torch.norm(stage1_end_pos - stage2_start_pos).item()
    if pos_gap_1_2 > 1e-2:
        warnings.append(f"Stage 1->2 position gap: {pos_gap_1_2:.6f}")

    stage2_end_pos = pred_traj2["pred_pos"][-1]
    stage3_start_pos = traj3_input_stitched["curr_pos"][0]
    pos_gap_2_3 = torch.norm(stage2_end_pos - stage3_start_pos).item()
    if pos_gap_2_3 > 1e-2:
        warnings.append(f"Stage 2->3 position gap: {pos_gap_2_3:.6f}")

    if "pred_stress" in pred_traj1 and "pred_stress" in pred_traj2:
        stage1_end_stress = pred_traj1["pred_stress"][-1]
        stage2_start_stress = pred_traj2["pred_stress"][0]
        stress_gap = torch.norm(stage1_end_stress - stage2_start_stress).item()
        if stress_gap > 0.1:
            warnings.append(f"Stage 1->2 stress jump: {stress_gap:.6f}")
    return warnings


def obtain_infer_traj(inference_config, checkpoint_path_1, checkpoint_path_2, checkpoint_path_3,
                      val_file_path, continuous, split_type="val"):
    params = PARAMETERS[inference_config["model"]]
    model1, model2, model3 = load_model(
        inference_config, params, checkpoint_path_1, checkpoint_path_2, checkpoint_path_3)

    print(f"\nPreparing DataLoaders for file: {val_file_path}")
    print(f"Mode: {split_type.upper()}")

    ds1 = TrajectoryDataset(val_file_path, split=split_type, stage=1)
    ds2 = TrajectoryDataset(val_file_path, split=split_type, stage=2)
    ds3 = TrajectoryDataset(val_file_path, split=split_type, stage=3)

    loader1 = DataLoader(ds1, batch_size=1, shuffle=False)
    loader2 = DataLoader(ds2, batch_size=1, shuffle=False)
    loader3 = DataLoader(ds3, batch_size=1, shuffle=False)

    validate_stage_alignment(loader1, loader2, loader3)

    trajectories1, trajectories2, trajectories3 = [], [], []
    info = {
        "index": [],
        "inference_time": [],
        "stage_1_time": [],
        "stage_2_time": [],
        "stage_3_time": [],
        "continuity_warnings": [],
    }

    print(f"Starting inference on {len(loader1)} trajectories...")
    count = 0

    def move_to_cpu(trajectory_dict):
        cpu_dict = {}
        for k, v in trajectory_dict.items():
            if isinstance(v, torch.Tensor):
                cpu_dict[k] = v.detach().cpu()
            else:
                cpu_dict[k] = v
        return cpu_dict

    with torch.no_grad():
        for batch_stage1, batch_stage2, batch_stage3 in zip(loader1, loader2, loader3):
            traj1_input = squeeze_data(batch_stage1)
            traj2_input = squeeze_data(batch_stage2)
            traj3_input = squeeze_data(batch_stage3)

            info["index"].append(count)
            print(f"Processing {split_type} Trajectory {count}...", end=" ")

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            if device.type == "cuda":
                torch.cuda.synchronize()
            stage1_start = time.perf_counter()
            _, pred_traj1 = params["evaluator"].evaluate(model1, traj1_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            stage1_time = time.perf_counter() - stage1_start
            info["stage_1_time"].append(stage1_time)

            if device.type == "cuda":
                torch.cuda.synchronize()
            stage2_start = time.perf_counter()
            if continuous:
                traj2_input_stitched = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in traj2_input.items()
                }
                traj2_input_stitched["curr_pos"][0] = pred_traj1["pred_pos"][-1]
                _, pred_traj2 = params["evaluator"].evaluate(model2, traj2_input_stitched)
                pred_traj2["gt_pos"][0] = traj2_input["curr_pos"][0]
            else:
                _, pred_traj2 = params["evaluator"].evaluate(model2, traj2_input)
                traj2_input_stitched = traj2_input
            if device.type == "cuda":
                torch.cuda.synchronize()
            stage2_time = time.perf_counter() - stage2_start
            info["stage_2_time"].append(stage2_time)

            if device.type == "cuda":
                torch.cuda.synchronize()
            stage3_start = time.perf_counter()
            if continuous:
                traj3_input_stitched = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in traj3_input.items()
                }
                traj3_input_stitched["curr_pos"][0] = pred_traj2["pred_pos"][-1]
                _, pred_traj3 = params["evaluator"].evaluate(model3, traj3_input_stitched)
                pred_traj3["gt_pos"][0] = traj3_input["curr_pos"][0]
            else:
                _, pred_traj3 = params["evaluator"].evaluate(model3, traj3_input)
                traj3_input_stitched = traj3_input
            if device.type == "cuda":
                torch.cuda.synchronize()
            stage3_time = time.perf_counter() - stage3_start
            info["stage_3_time"].append(stage3_time)

            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time = time.perf_counter() - start_time

            warnings = validate_stage_continuity(
                pred_traj1, pred_traj2, pred_traj3,
                traj2_input_stitched, traj3_input_stitched)
            if warnings:
                for w in warnings:
                    print(f"\n  {w}")
                info["continuity_warnings"].extend(warnings)

            trajectories1.append(move_to_cpu(pred_traj1))
            trajectories2.append(move_to_cpu(pred_traj2))
            trajectories3.append(move_to_cpu(pred_traj3))

            info["inference_time"].append(total_time)
            print(f"({total_time*1000:.2f} ms)")
            count += 1

    return trajectories1, trajectories2, trajectories3, info


def save_rollout(rollout_path, trajectories, disp=""):
    with open(rollout_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"  Saved {disp} rollout to: {rollout_path}")


def concat_trajectories(trajectories1, trajectories2, trajectories3):
    num_samples_1 = len(trajectories1)
    num_samples_2 = len(trajectories2)
    num_samples_3 = len(trajectories3)
    if not (num_samples_1 == num_samples_2 == num_samples_3):
        raise ValueError(
            f"Stage sample count mismatch:\n"
            f"  Stage 1: {num_samples_1} samples\n"
            f"  Stage 2: {num_samples_2} samples\n"
            f"  Stage 3: {num_samples_3} samples"
        )

    concatenated_results = []
    for i in range(num_samples_1):
        sample_dict = {}
        keys = trajectories1[i].keys()
        for key in keys:
            t1 = trajectories1[i][key]
            t2 = trajectories2[i][key]
            t3 = trajectories3[i][key]
            if not isinstance(t1, torch.Tensor):
                sample_dict[key] = t1
                continue
            try:
                sample_dict[key] = torch.cat([t1, t2, t3], dim=0)
            except RuntimeError as e:
                print(f"Warning: Could not concatenate key '{key}' for sample {i}")
                print(f"     Shapes: Stage1={t1.shape}, Stage2={t2.shape}, Stage3={t3.shape}")
                print(f"     Error: {e}")
                sample_dict[key] = t1
        concatenated_results.append(sample_dict)
    return concatenated_results


def inference(checkpoint_path_1, checkpoint_path_2, checkpoint_path_3, val_file_path,
              rollout_dir, continuous=True, evaluate=False, animation=False,
              animation_html=False, split_type="val", core_model="encode_process_decode"):
    inference_config = {
        "model": "press",
        "mode": "all",
        "core_model": core_model,
        "aggregator": "sum",
        "steps": 16,
        "attention": False,
    }

    print("\n" + "=" * 70)
    print("STARTING 3-STAGE INFERENCE PIPELINE")
    print(f"Input: {val_file_path}")
    print(f"Split: {split_type.upper()}")
    print(f"Continuous Mode: {'ON (stages stitched)' if continuous else 'OFF (independent)'}")
    print("=" * 70)

    trajs1, trajs2, trajs3, info = obtain_infer_traj(
        inference_config, checkpoint_path_1, checkpoint_path_2, checkpoint_path_3,
        val_file_path, continuous, split_type
    )

    os.makedirs(rollout_dir, exist_ok=True)
    print(f"\nSaving individual stages to: {rollout_dir}")
    with open(os.path.join(rollout_dir, "rollout_info.json"), "w") as f:
        info_serializable = {
            "index": info["index"],
            "inference_time": [float(x) for x in info["inference_time"]],
            "stage_1_time": [float(x) for x in info["stage_1_time"]],
            "stage_2_time": [float(x) for x in info["stage_2_time"]],
            "stage_3_time": [float(x) for x in info["stage_3_time"]],
            "continuity_warnings": info["continuity_warnings"],
        }
        json.dump(info_serializable, f, indent=4)

    save_rollout(os.path.join(rollout_dir, "rollout_stage_1.pkl"), trajs1, disp="Stage 1 (Press)")
    save_rollout(os.path.join(rollout_dir, "rollout_stage_2.pkl"), trajs2, disp="Stage 2 (Dwell)")
    save_rollout(os.path.join(rollout_dir, "rollout_stage_3.pkl"), trajs3, disp="Stage 3 (Release)")

    print(f"\nConcatenating {len(trajs1)} trajectories from 3 stages...")
    combined_trajs = concat_trajectories(trajs1, trajs2, trajs3)
    print(f"Concatenation complete. Output shape: {combined_trajs[0]['pred_pos'].shape}")

    del trajs1
    del trajs2
    del trajs3
    gc.collect()

    output_file = os.path.join(rollout_dir, "concatenated_rollout_all.pkl")
    save_rollout(output_file, combined_trajs, disp="Concatenated All Stages")

    print(f"\nTiming Statistics:")
    print(f"  Stage 1 (Press):   {sum(info['stage_1_time']):.2f}s avg: {sum(info['stage_1_time'])/len(info['stage_1_time']):.4f}s")
    print(f"  Stage 2 (Dwell):   {sum(info['stage_2_time']):.2f}s avg: {sum(info['stage_2_time'])/len(info['stage_2_time']):.4f}s")
    print(f"  Stage 3 (Release): {sum(info['stage_3_time']):.2f}s avg: {sum(info['stage_3_time'])/len(info['stage_3_time']):.4f}s")
    print(f"  Total Inference:   {sum(info['inference_time']):.2f}s avg: {sum(info['inference_time'])/len(info['inference_time']):.4f}s")

    if evaluate:
        try:
            from pressnetpp.utilities.evaluation_stress import evaluate_rollout
            evaluate_dir = os.path.join(rollout_dir, "rollout_evaluation")
            print(f"\nRunning Evaluation for {split_type}...")
            evaluate_rollout(output_file, evaluate_dir, plot=True)
        except ImportError:
            print("Evaluation module not found, skipping.")

    if animation:
        try:
            from pressnetpp.utilities.animate_rollout import animate_rollout
            vis_root = os.path.join(rollout_dir, "visualization")
            os.makedirs(vis_root, exist_ok=True)
            print(f"\nGenerating Video Animations for {split_type}...")
            for i in range(len(combined_trajs)):
                traj_vis_path = os.path.join(vis_root, f"{i}")
                os.makedirs(traj_vis_path, exist_ok=True)
                animate_rollout(output_file, traj_vis_path, i=i, key="stress")
        except ImportError:
            print("Animation module not found, skipping.")

    if animation_html:
        try:
            from pressnetpp.utilities.animate_compare import animate_rollout as animate_rollout_html
            vis_root = os.path.join(rollout_dir, "visualization")
            os.makedirs(vis_root, exist_ok=True)
            print(f"\nGenerating HTML Visualization for {split_type}...")
            animate_rollout_html(output_file, vis_root)
        except ImportError:
            print("HTML Animation module not found, skipping.")

    print(f"\n" + "=" * 70)
    print(f"INFERENCE COMPLETE for {split_type.upper()}")
    print("=" * 70 + "\n")
    return


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Run 3-stage Pressnet++ inference.")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to JSON config file (default: config.json)")
    parser.add_argument("--split", type=str, default="val", choices=["test", "val", "both"],
                        help="Which dataset split to run inference on (default: val)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve config and print, then exit without inference")
    return parser


def main():
    parser = _build_arg_parser()
    args, _ = parser.parse_known_args()

    config = {}
    try:
        with open(args.config, "r") as f:
            config = json.load(f)
        print(f"Loaded config from {args.config}")
    except FileNotFoundError:
        print(f"Config file {args.config} not found. Using default argument values.")
    except json.JSONDecodeError:
        print(f"Error parsing {args.config}. Using default argument values.")

    parser.add_argument("--checkpoint_path_1", type=str,
                        default=config.get("checkpoint_path_1") or None,
                        help="Checkpoint path for Stage 1 (Press)")
    parser.add_argument("--checkpoint_path_2", type=str,
                        default=config.get("checkpoint_path_2") or None,
                        help="Checkpoint path for Stage 2 (Dwell)")
    parser.add_argument("--checkpoint_path_3", type=str,
                        default=config.get("checkpoint_path_3") or None,
                        help="Checkpoint path for Stage 3 (Release)")
    parser.add_argument("--val_file", type=str,
                        default=config.get("val_file",
                                           "/home/sushil/PressNet/datasets/data/coarse/coarse_1500_train_val.h5"),
                        help="HDF5 file to run inference on")
    parser.add_argument("--rollout_dir", type=str,
                        default=config.get("rollout_dir",
                                           "/home/sushil/PressNet/Pressnet++/output/inference"),
                        help="Directory to save rollout pkl files")
    parser.add_argument("--core_model", type=str,
                        default=config.get("core_model", "encode_process_decode"),
                        help="Core model name(s): str, list of 3, or dict stage_1/2/3")
    parser.add_argument("--continuous", action="store_true",
                        default=config.get("continuous", True),
                        help="Stitch stages in continuous mode")
    parser.add_argument("--evaluate", action="store_true",
                        default=config.get("evaluate", False),
                        help="Run rollout evaluation after inference")
    parser.add_argument("--animation", action="store_true",
                        default=config.get("animation", False),
                        help="Generate video animations after inference")
    parser.add_argument("--animation_html", action="store_true",
                        default=config.get("animation_html", False),
                        help="Generate HTML visualizations after inference")

    args = parser.parse_args()

    print(f"Core model(s): {args.core_model}")
    print(f"Val file: {args.val_file}")
    print(f"Rollout dir: {args.rollout_dir}")

    if args.dry_run:
        print("\n[DRY RUN] Configuration resolved. Exiting without inference.")
        return

    missing = [name for name in ("checkpoint_path_1", "checkpoint_path_2", "checkpoint_path_3")
               if not getattr(args, name)]
    if missing:
        parser.error("Missing required arguments: " + ", ".join(
            "--" + name for name in missing))

    all_tasks = [
        {"split": "val", "file": args.val_file},
    ]
    if args.split == "both":
        all_tasks.append({"split": "test", "file": args.val_file})
    elif args.split == "test":
        all_tasks = [{"split": "test", "file": args.val_file}]

    for task in all_tasks:
        split_name = task["split"]
        file_path = task["file"]
        if not os.path.exists(file_path):
            print(f"\n[SKIP] File not found: {file_path}")
            continue
        current_rollout_dir = os.path.join(args.rollout_dir, split_name)
        print(f"\n{'='*70}")
        print(f" STARTING TASK: {split_name.upper()}")
        print(f" Input File: {file_path}")
        print(f" Output Dir: {current_rollout_dir}")
        print(f"{'='*70}")
        inference(
            checkpoint_path_1=args.checkpoint_path_1,
            checkpoint_path_2=args.checkpoint_path_2,
            checkpoint_path_3=args.checkpoint_path_3,
            val_file_path=file_path,
            rollout_dir=current_rollout_dir,
            continuous=args.continuous,
            evaluate=args.evaluate,
            animation=args.animation,
            animation_html=args.animation_html,
            split_type=split_name,
            core_model=args.core_model,
        )


if __name__ == "__main__":
    main()