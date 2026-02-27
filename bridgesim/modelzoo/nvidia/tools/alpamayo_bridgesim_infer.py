import argparse
import json
from pathlib import Path

import numpy as np
import torch

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper


def yaw_to_rotmat(yaw: np.ndarray) -> np.ndarray:
    """
    yaw: (...,) radians
    returns: (..., 3, 3)
    """
    c = np.cos(yaw).astype(np.float32)
    s = np.sin(yaw).astype(np.float32)
    R = np.zeros(yaw.shape + (3, 3), dtype=np.float32)
    R[..., 0, 0] = c
    R[..., 0, 1] = -s
    R[..., 1, 0] = s
    R[..., 1, 1] = c
    R[..., 2, 2] = 1.0
    return R


def world_to_ego_local(ego_xyz: np.ndarray, ego_rotmats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert world-frame history to ego-local frame at t0 (last step).
    """
    t0_xyz = ego_xyz[:, :, -1, :]          # (1,1,3)
    t0_R = ego_rotmats[:, :, -1, :, :]     # (1,1,3,3)
    t0_R_inv = np.swapaxes(t0_R, -1, -2)   # transpose (rotation inverse)

    dxyz = ego_xyz - t0_xyz[:, :, None, :]  # (1,1,T,3)

    xyz_local = np.einsum("...ij,...tj->...ti", t0_R_inv, dxyz).astype(np.float32)
    rot_local = np.einsum("...ij,...tjk->...tik", t0_R_inv, ego_rotmats).astype(np.float32)
    return xyz_local, rot_local


def pad_or_trim_time(x: np.ndarray, required_T: int) -> np.ndarray:
    """
    x has shape (1,1,T,...) where time dim is index 2
    - If T > required_T: keep most recent required_T
    - If T < required_T: left-pad by repeating earliest frame
    """
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
    """
    Alpamayo helper expects frames: (N, C, H, W) uint8.
    """
    img = np.load(image_npy)

    # normalize to (N, H, W, 3)
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

    # convert HWC -> CHW
    img = np.transpose(img, (0, 3, 1, 2))
    img = np.ascontiguousarray(img)

    return torch.from_numpy(img)  # uint8 CPU (N,3,H,W)


def load_ego_history(ego_xyz_npy: str, ego_rot_npy: str, required_T: int = 16):
    """
    Load ego history arrays from .npy files and normalize to Alpamayo's expected history length.

    Supports BOTH ego_rot formats:
      - heading/yaw: (1,1,T) or (1,1,T,1)   (radians)
      - rotmats:     (1,1,T,3,3)
    """
    ego_xyz = np.load(ego_xyz_npy)
    ego_rot = np.load(ego_rot_npy)

    if ego_xyz.ndim != 4 or ego_xyz.shape[:2] != (1, 1) or ego_xyz.shape[-1] != 3:
        raise ValueError(f"Expected ego_xyz shape (1,1,T,3). Got {ego_xyz.shape}")
    T = ego_xyz.shape[2]

    # rot handling
    if ego_rot.ndim == 5 and ego_rot.shape[:3] == (1, 1, T) and ego_rot.shape[-2:] == (3, 3):
        ego_rotmats = ego_rot.astype(np.float32)
    elif ego_rot.ndim == 4 and ego_rot.shape[:3] == (1, 1, T) and ego_rot.shape[-1] == 1:
        yaw = ego_rot[..., 0].astype(np.float32)  # (1,1,T)
        ego_rotmats = yaw_to_rotmat(yaw)          # (1,1,T,3,3)
    elif ego_rot.ndim == 3 and ego_rot.shape == (1, 1, T):
        yaw = ego_rot.astype(np.float32)
        ego_rotmats = yaw_to_rotmat(yaw)
    else:
        raise ValueError(
            "Expected ego_rot as rotmats (1,1,T,3,3) or heading (1,1,T,1)/(1,1,T). "
            f"Got {ego_rot.shape}"
        )

    # normalize T
    ego_xyz = pad_or_trim_time(ego_xyz.astype(np.float32), required_T)
    ego_rotmats = pad_or_trim_time(ego_rotmats.astype(np.float32), required_T)

    # convert world -> ego-local at t0
    ego_xyz, ego_rotmats = world_to_ego_local(ego_xyz, ego_rotmats)

    ego_history_xyz = torch.from_numpy(ego_xyz).float()
    ego_history_rot = torch.from_numpy(ego_rotmats).float()
    return ego_history_xyz, ego_history_rot


def inject_nav_command(messages, nav_cmd: str):
    """
    Put NAV_COMMAND into the last user text block.
    """
    user_content = messages[1]["content"]
    for i in range(len(user_content) - 1, -1, -1):
        if user_content[i].get("type") == "text":
            user_content[i]["text"] = f"NAV_COMMAND: {nav_cmd}\n" + user_content[i]["text"]
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--image-npy", required=True, type=str)
    parser.add_argument("--ego-xyz-npy", required=True, type=str)
    parser.add_argument("--ego-rot-npy", required=True, type=str)
    parser.add_argument("--out-json", required=True, type=str)

    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--max-generation-length", type=int, default=256)

    parser.add_argument("--nav-cmd", type=str, default="STRAIGHT")
    parser.add_argument("--debug-print-shapes", action="store_true")

    parser.add_argument(
        "--coord-mode",
        type=str,
        default="x_forward_y_left",
        choices=[
            "x_forward_y_left",     # forward=x, lateral(left)=y
            "x_forward_y_right",    # forward=x, lateral(left)=-y
            "x_right_y_forward",    # forward=y, lateral(left)=-x
            "x_left_y_forward",     # forward=y, lateral(left)=x
        ],
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Load model
    model = AlpamayoR1.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    # Inputs
    frames = load_frames_as_expected(args.image_npy, target_n=16)  # (16,3,H,W) uint8 CPU
    messages = helper.create_message(frames)
    inject_nav_command(messages, args.nav_cmd)

    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    ego_history_xyz, ego_history_rot = load_ego_history(
        args.ego_xyz_npy, args.ego_rot_npy, required_T=16
    )

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    model_inputs = helper.to_device(model_inputs, device)

    if args.debug_print_shapes:
        print("tokenized_data keys:", list(inputs.keys()))
        for k, v in inputs.items():
            if hasattr(v, "shape"):
                print(k, tuple(v.shape), v.dtype, v.device)
        print("ego_history_xyz", tuple(model_inputs["ego_history_xyz"].shape), model_inputs["ego_history_xyz"].dtype, model_inputs["ego_history_xyz"].device)
        print("ego_history_rot", tuple(model_inputs["ego_history_rot"].shape), model_inputs["ego_history_rot"].dtype, model_inputs["ego_history_rot"].device)

    if device == "cuda":
        torch.cuda.manual_seed_all(42)

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=args.top_p,
                temperature=args.temperature,
                num_traj_samples=args.num_traj_samples,
                max_generation_length=args.max_generation_length,
                return_extra=True,
            )

    # Convert to (T,2) lateral/forward
    xyz = pred_xyz.detach().float().cpu().numpy()
    # expected: [B, num_traj_sets, num_traj_samples, T, 3]
    if xyz.ndim == 5:
        traj_xyz = xyz[0, 0, 0]
    else:
        traj_xyz = xyz.reshape(-1, xyz.shape[-1])

    if traj_xyz.ndim != 2 or traj_xyz.shape[1] < 2:
        traj_lat_fwd = [[0.0, 0.0] for _ in range(10)]
    else:
        # Remove any absolute offset so controller sees a plan starting at 0,0
        traj_xyz = traj_xyz - traj_xyz[0:1]

        x = traj_xyz[:, 0]
        y = traj_xyz[:, 1]

        # Map model XY into (forward, lateral-left) depending on coord-mode
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

    out = {
        "trajectory_lateral_forward": traj_lat_fwd,
        "reasoning": reasoning,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))


if __name__ == "__main__":
    main()