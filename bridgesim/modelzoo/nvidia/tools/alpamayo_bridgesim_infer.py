"""
Persistent inference server for Alpamayo-R1.

Launched once by AlpamayoSubprocessClient. Loads the model, prints "READY" to
stdout, then loops reading JSON request lines from stdin and writing JSON result
lines to stdout.

Request  (stdin, one JSON line per frame):
    {"image_npy": "<path>", "ego_xyz_npy": "<path>", "ego_rot_npy": "<path>", "nav_cmd": "<str>"}

Response (stdout, one JSON line per frame):
    {"trajectory_lateral_forward": [[lat, fwd], ...], "reasoning": "<str|null>"}
    {"error": "<msg>"}   -- on failure

All model-level params (top_p, temperature, coord_mode, …) are fixed at startup
via CLI args.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper


# ---------------------------------------------------------------------------
# Geometry helpers (unchanged from original)
# ---------------------------------------------------------------------------

def yaw_to_rotmat(yaw: np.ndarray) -> np.ndarray:
    c = np.cos(yaw).astype(np.float32)
    s = np.sin(yaw).astype(np.float32)
    R = np.zeros(yaw.shape + (3, 3), dtype=np.float32)
    R[..., 0, 0] = c
    R[..., 0, 1] = -s
    R[..., 1, 0] = s
    R[..., 1, 1] = c
    R[..., 2, 2] = 1.0
    return R


def world_to_ego_local(ego_xyz: np.ndarray, ego_rotmats: np.ndarray):
    t0_xyz = ego_xyz[:, :, -1, :]
    t0_R = ego_rotmats[:, :, -1, :, :]
    t0_R_inv = np.swapaxes(t0_R, -1, -2)
    dxyz = ego_xyz - t0_xyz[:, :, None, :]
    xyz_local = np.einsum("...ij,...tj->...ti", t0_R_inv, dxyz).astype(np.float32)
    rot_local = np.einsum("...ij,...tjk->...tik", t0_R_inv, ego_rotmats).astype(np.float32)
    return xyz_local, rot_local


def pad_or_trim_time(x: np.ndarray, required_T: int) -> np.ndarray:
    T_cur = x.shape[2]
    if T_cur == required_T:
        return x
    if T_cur > required_T:
        return x[:, :, -required_T:, ...]
    pad_n = required_T - T_cur
    first = x[:, :, 0:1, ...]
    pad = np.repeat(first, pad_n, axis=2)
    return np.concatenate([pad, x], axis=2)


def load_frames_as_expected(image_npy: str, target_n: int = 16) -> torch.Tensor:
    img = np.load(image_npy)
    if img.ndim == 3:
        img = img[None, ...]
    if img.ndim != 4 or img.shape[-1] != 3:
        raise ValueError(f"Expected image npy as (H,W,3) or (N,H,W,3). Got {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[0] < target_n:
        reps = target_n - img.shape[0]
        img = np.concatenate([img[:1]] * reps + [img], axis=0)
    elif img.shape[0] > target_n:
        img = img[-target_n:]
    img = np.transpose(img, (0, 3, 1, 2))
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img)


def load_ego_history(ego_xyz_npy: str, ego_rot_npy: str, required_T: int = 16):
    ego_xyz = np.load(ego_xyz_npy)
    ego_rot = np.load(ego_rot_npy)

    if ego_xyz.ndim != 4 or ego_xyz.shape[:2] != (1, 1) or ego_xyz.shape[-1] != 3:
        raise ValueError(f"Expected ego_xyz shape (1,1,T,3). Got {ego_xyz.shape}")
    T = ego_xyz.shape[2]

    if ego_rot.ndim == 5 and ego_rot.shape[:3] == (1, 1, T) and ego_rot.shape[-2:] == (3, 3):
        ego_rotmats = ego_rot.astype(np.float32)
    elif ego_rot.ndim == 4 and ego_rot.shape[:3] == (1, 1, T) and ego_rot.shape[-1] == 1:
        yaw = ego_rot[..., 0].astype(np.float32)
        ego_rotmats = yaw_to_rotmat(yaw)
    elif ego_rot.ndim == 3 and ego_rot.shape == (1, 1, T):
        yaw = ego_rot.astype(np.float32)
        ego_rotmats = yaw_to_rotmat(yaw)
    else:
        raise ValueError(
            "Expected ego_rot as rotmats (1,1,T,3,3) or heading (1,1,T,1)/(1,1,T). "
            f"Got {ego_rot.shape}"
        )

    ego_xyz = pad_or_trim_time(ego_xyz.astype(np.float32), required_T)
    ego_rotmats = pad_or_trim_time(ego_rotmats.astype(np.float32), required_T)
    ego_xyz, ego_rotmats = world_to_ego_local(ego_xyz, ego_rotmats)
    return torch.from_numpy(ego_xyz).float(), torch.from_numpy(ego_rotmats).float()


def inject_nav_command(messages, nav_cmd: str):
    user_content = messages[1]["content"]
    for i in range(len(user_content) - 1, -1, -1):
        if user_content[i].get("type") == "text":
            user_content[i]["text"] = f"NAV_COMMAND: {nav_cmd}\n" + user_content[i]["text"]
            break


# ---------------------------------------------------------------------------
# Single-frame inference
# ---------------------------------------------------------------------------

# Waypoint indices for subsampling 10Hz output to 8 waypoints at 0.5s intervals
# Index i corresponds to t=(i+1)*0.1s, so index 4=0.5s, 9=1.0s, ..., 39=4.0s
_PLANNER_WAYPOINT_INDICES = [4, 9, 14, 19, 24, 29, 34, 39]


def _xyz_to_fwd_lat_z(all_xyz: np.ndarray, coord_mode: str) -> np.ndarray:
    """Convert (N, T, 3) xyz array to (N, T, 3) [fwd, lat, z] for scorer convention."""
    x, y, z = all_xyz[:, :, 0], all_xyz[:, :, 1], all_xyz[:, :, 2]
    if coord_mode == "x_forward_y_left":
        fwd, lat = x, y
    elif coord_mode == "x_forward_y_right":
        fwd, lat = x, -y
    elif coord_mode == "x_right_y_forward":
        fwd, lat = y, -x
    elif coord_mode == "x_left_y_forward":
        fwd, lat = y, x
    else:
        fwd, lat = x, y
    return np.stack([fwd, lat, z], axis=-1)


def run_one(model, processor, device, req: dict, args) -> dict:
    nav_cmd = req.get("nav_cmd", "STRAIGHT")
    num_inference_groups = req.get("num_inference_groups", 1)
    total_samples = args.num_traj_samples * num_inference_groups

    frames = load_frames_as_expected(req["image_npy"], target_n=16)
    messages = helper.create_message(frames)
    inject_nav_command(messages, nav_cmd)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    ego_history_xyz, ego_history_rot = load_ego_history(
        req["ego_xyz_npy"], req["ego_rot_npy"], required_T=16
    )

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    model_inputs = helper.to_device(model_inputs, device)

    if device == "cuda" and total_samples == 1:
        torch.cuda.manual_seed_all(42)

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=args.top_p,
                temperature=args.temperature,
                num_traj_samples=total_samples,
                max_generation_length=args.max_generation_length,
                return_extra=True,
            )

    xyz = pred_xyz.detach().float().cpu().numpy()
    print(f"[ALPAMAYO DEBUG] pred_xyz shape: {xyz.shape}, total_samples={total_samples}", file=sys.stderr, flush=True)
    if xyz.ndim == 5:
        all_xyz = xyz[0, 0]  # (N, T, 3)
    else:
        all_xyz = xyz.reshape(total_samples, -1, xyz.shape[-1])
    print(f"[ALPAMAYO DEBUG] all_xyz shape: {all_xyz.shape}, unique rows: {len(set(tuple(all_xyz[i,4]) for i in range(len(all_xyz))))}", file=sys.stderr, flush=True)

    traj_xyz = all_xyz[0]  # (T, 3) — first sample for single-traj output

    if traj_xyz.ndim != 2 or traj_xyz.shape[1] < 2:
        traj_lat_fwd = [[0.0, 0.0] for _ in range(10)]
    else:
        traj_xyz = traj_xyz - traj_xyz[0:1]
        x = traj_xyz[:, 0]
        y = traj_xyz[:, 1]
        if args.coord_mode == "x_forward_y_left":
            forward, lateral = x, y
        elif args.coord_mode == "x_forward_y_right":
            forward, lateral = x, -y
        elif args.coord_mode == "x_right_y_forward":
            forward, lateral = y, -x
        elif args.coord_mode == "x_left_y_forward":
            forward, lateral = y, x
        else:
            forward, lateral = x, y
        traj_lat_fwd = np.stack([lateral, forward], axis=1).tolist()

    reasoning = None
    try:
        if isinstance(extra, dict) and "cot" in extra:
            cot = extra["cot"]
            reasoning = str(cot[0]) if hasattr(cot, "__len__") else str(cot)
    except Exception:
        reasoning = None

    result = {"trajectory_lateral_forward": traj_lat_fwd, "reasoning": reasoning}

    # Build all_candidates for inference scaling (always when num_inference_groups in req)
    if "num_inference_groups" in req:
        T = all_xyz.shape[1]
        # Subtract per-sample origin so waypoints are ego-relative offsets
        origins = all_xyz[:, 0:1, :]  # (N, 1, 3)
        all_xyz_rel = all_xyz - origins  # (N, T, 3)
        all_xyz_rel = _xyz_to_fwd_lat_z(all_xyz_rel, args.coord_mode)  # (N, T, 3) [fwd,lat,z]

        # Coarse candidates at 0.5s intervals for scorer (N, 8, 3)
        indices = [i for i in _PLANNER_WAYPOINT_INDICES if i < T]
        while len(indices) < len(_PLANNER_WAYPOINT_INDICES):
            indices.append(indices[-1])
        result["all_candidates"] = all_xyz_rel[:, indices, :].tolist()  # (N, 8, 3)

        # Full-resolution candidates at 10Hz for evaluator consumption after scorer picks best
        result["all_candidates_full"] = all_xyz_rel.tolist()  # (N, T, 3)

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument(
        "--coord-mode",
        type=str,
        default="x_forward_y_left",
        choices=["x_forward_y_left", "x_forward_y_right", "x_right_y_forward", "x_left_y_forward"],
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[alpamayo_server] Loading {args.model} ...", file=sys.stderr, flush=True)
    model = AlpamayoR1.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    print("[alpamayo_server] Model ready.", file=sys.stderr, flush=True)

    # Signal the parent that we are ready to accept requests.
    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            result = run_one(model, processor, device, req, args)
        except Exception as e:
            import traceback
            result = {"error": str(e), "traceback": traceback.format_exc()}
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
