"""
OpenPilot model adapter for BridgeSim evaluation.

Self-contained implementation — all necessary constants, warp computation,
YUV conversion, and output parsing are reimplemented in-file (no OpenPilot imports).

OpenPilot's model is a two-stage architecture:
  1. Vision model (CNN): processes camera images → features
  2. Policy model (temporal): accumulated features → trajectory plan
The vision model gets fed 4 frames (2 from narrow, 2 from wide) 0.2s apart.
The policy model takes in up to 25 frames of vision features 0.2s apart (up to 5s temporal context).

Both stages are ONNX models run via onnxruntime.
"""

import collections
import pickle
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import onnxruntime as ort

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter

# ---------------------------------------------------------------------------
# Constants copied from openpilot repo
# ---------------------------------------------------------------------------

# 33 non-uniform time points, 0–10 s (quadratic spacing)
T_IDXS = [10.0 * (i / 32) ** 2 for i in range(33)]

PLAN_MHP_N = 5     # number of trajectory hypotheses
PLAN_WIDTH = 15    # pos(3) + vel(3) + accel(3) + euler(3) + orient_rate(3)
IDX_N = 33
FEATURE_LEN = 512
DESIRE_LEN = 8

# Model input image size after warp
MEDMODEL_INPUT_SIZE = (512, 256)  # (W, H)

# Model-space intrinsics (what the model "sees")
MEDMODEL_FL = 910.0
MEDMODEL_CY = 47.6
SBIGMODEL_FL = 455.0
SBIGMODEL_CY = 151.8

# view_frame_from_device_frame  (device: x=fwd, y=right, z=down →
#                                  view:   x=right, y=down, z=fwd)
VIEW_FRAME_FROM_DEVICE = np.array(
    [[0, 0, 1],
     [1, 0, 0],
     [0, 1, 0]], dtype=np.float64
).T  # transpose gives device→view


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def rot_from_euler(rpy):
    """3×3 rotation matrix from roll/pitch/yaw (ZYX convention)."""
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _build_model_intrinsics():
    """Build intrinsic matrices for med-model and sbig-model."""
    med = np.array([
        [MEDMODEL_FL, 0, MEDMODEL_INPUT_SIZE[0] / 2.0],
        [0, MEDMODEL_FL, MEDMODEL_CY],
        [0, 0, 1],
    ], dtype=np.float64)

    sbig = np.array([
        [SBIGMODEL_FL, 0, MEDMODEL_INPUT_SIZE[0] / 2.0],
        [0, SBIGMODEL_FL, SBIGMODEL_CY],
        [0, 0, 1],
    ], dtype=np.float64)
    return med, sbig


def _build_calib_from_model(model_intrinsics):
    """
    Compute calib_from_model = inv(model_intrinsics @ view_from_device @ I).
    With zero calibration (identity device_from_calib), this simplifies to
    inv(model_intrinsics @ view_from_device).
    """
    M = model_intrinsics @ VIEW_FRAME_FROM_DEVICE  # 3×3
    return np.linalg.inv(M)


def get_warp_matrix(device_from_calib_euler, camera_intrinsics, bigmodel_frame=False):
    """
    Compute the 3×3 warp matrix for cv2.warpPerspective.

    Maps model-space pixels → camera pixels so that warpPerspective samples
    the camera image at the correct locations.
    """
    med_intr, sbig_intr = _build_model_intrinsics()
    calib_from_model = _build_calib_from_model(sbig_intr if bigmodel_frame else med_intr)
    device_from_calib = rot_from_euler(device_from_calib_euler)
    camera_from_calib = camera_intrinsics @ VIEW_FRAME_FROM_DEVICE @ device_from_calib
    return camera_from_calib @ calib_from_model


def rgb_to_yuv6ch(rgb_512x256):
    """
    Convert warped 512×256 RGB image to 6-channel YUV420 at (6, 128, 256).

    Uses BT.601 integer coefficients matching OpenPilot's camerad.py.
    """
    r = rgb_512x256[:, :, 0].astype(np.int32)
    g = rgb_512x256[:, :, 1].astype(np.int32)
    b = rgb_512x256[:, :, 2].astype(np.int32)

    # Y plane (256, 512)
    Y = np.clip((((b * 13 + g * 65 + r * 33) + 64) >> 7) + 16, 0, 255).astype(np.uint8)

    # Y decomposition into 4 half-res channels (128, 256)
    # Order must match OpenPilot's frames_to_tensor: even/even, odd/even, even/odd, odd/odd
    ch0 = Y[0::2, 0::2]  # even row, even col
    ch1 = Y[1::2, 0::2]  # odd row, even col
    ch2 = Y[0::2, 1::2]  # even row, odd col
    ch3 = Y[1::2, 1::2]  # odd row, odd col

    # 2×2 box-filter sub-sample for chroma
    r_sub = (r[0::2, 0::2] + r[0::2, 1::2] + r[1::2, 0::2] + r[1::2, 1::2] + 2) >> 2
    g_sub = (g[0::2, 0::2] + g[0::2, 1::2] + g[1::2, 0::2] + g[1::2, 1::2] + 2) >> 2
    b_sub = (b[0::2, 0::2] + b[0::2, 1::2] + b[1::2, 0::2] + b[1::2, 1::2] + 2) >> 2

    U = np.clip((b_sub * 56 - g_sub * 37 - r_sub * 19 + 0x8080) >> 8, 0, 255).astype(np.uint8)
    V = np.clip((r_sub * 56 - g_sub * 47 - b_sub * 9 + 0x8080) >> 8, 0, 255).astype(np.uint8)

    return np.stack([ch0, ch1, ch2, ch3, U, V], axis=0)  # (6, 128, 256)


def safe_exp(x, clip_val=88.0):
    return np.exp(np.clip(x, -clip_val, clip_val))


def softmax(x, axis=-1):
    e = safe_exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def parse_mdn(raw, in_N, out_N, out_shape):
    """
    Mixture density network output parsing (reimplemented from OpenPilot).

    Args:
        raw: (1, total) raw output slice
        in_N: number of input hypotheses (0 = no MHP)
        out_N: number of output selections (0 = no MHP)
        out_shape: shape of each output (e.g. (33, 15))

    Returns:
        (1, *out_shape) parsed output (best hypothesis if MHP, else just mu)
    """
    raw = raw.reshape((raw.shape[0], max(in_N, 1), -1))
    n_values = (raw.shape[2] - out_N) // 2
    pred_mu = raw[:, :, :n_values]
    pred_std = safe_exp(raw[:, :, n_values:2 * n_values])

    if in_N > 1:
        # MHP mode: select best hypothesis by weight
        weights = np.zeros((raw.shape[0], in_N, out_N), dtype=raw.dtype)
        for i in range(out_N):
            weights[:, :, i - out_N] = softmax(raw[:, :, i - out_N], axis=-1)

        pred_mu_final = np.zeros((raw.shape[0], max(out_N, 1), n_values), dtype=raw.dtype)
        for fidx in range(weights.shape[0]):
            for hidx in range(out_N):
                idxs = np.argsort(weights[fidx, :, hidx])[::-1]
                pred_mu_final[fidx, hidx] = pred_mu[fidx, idxs[0]]
    else:
        pred_mu_final = pred_mu

    if out_N > 1:
        final_shape = tuple([raw.shape[0], out_N] + list(out_shape))
    else:
        final_shape = tuple([raw.shape[0]] + list(out_shape))
    return pred_mu_final.reshape(final_shape)


# ---------------------------------------------------------------------------
# Camera intrinsic helpers
# ---------------------------------------------------------------------------

def _fov_to_focal(fov_deg, pixel_width):
    """Compute focal length from horizontal FOV and image width."""
    return (pixel_width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)


def _camera_intrinsics(fov_deg, width, height):
    """Build 3×3 camera intrinsic matrix from FOV and resolution."""
    f = _fov_to_focal(fov_deg, width)
    cx, cy = width / 2.0, height / 2.0
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# OpenPilotAdapter
# ---------------------------------------------------------------------------

class OpenPilotAdapter(BaseModelAdapter):
    """
    Adapter for OpenPilot's driving model (vision + policy ONNX).
    """

    def __init__(self, checkpoint_path: str, **kwargs):
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.ckpt_dir = Path(checkpoint_path)

        # ONNX session handles
        self.vision_session = None
        self.policy_session = None

        # Output slice metadata (loaded from pickle)
        self.vision_meta = None
        self.policy_meta = None

        # Rolling frame buffers — one deque per camera, each stores (6,128,256) uint8
        self.road_frames = collections.deque(maxlen=3)
        self.wide_frames = collections.deque(maxlen=3)

        # Temporal buffers for policy model.
        # Architecture mirrors OpenPilot's modeld.py:
        #   - Vision runs every prepare_input call (10Hz), producing features
        #   - Features are stored in a raw buffer at 10Hz
        #   - Policy model (run_inference) subsamples to 5Hz (200ms spacing)
        #     for 25 time steps = 5s of temporal context
        # This matches OpenPilot's InputQueues(model_fps=5, env_fps=20) pattern
        # and works correctly for any replan_rate.
        self._steps_per_context = 2   # 10Hz / 5Hz = 2 steps per context slot
        self._raw_buffer_len = 50     # 25 context slots * 2 steps each

        # Raw buffers at 10Hz (subsampled to 5Hz for policy model)
        self._raw_features = np.zeros((1, self._raw_buffer_len, FEATURE_LEN), dtype=np.float16)
        self._raw_desire = np.zeros((1, self._raw_buffer_len, DESIRE_LEN), dtype=np.float16)

        # Desire tracking — mirrors OpenPilot's rising-edge pulse logic.
        # Rising edges are detected at 10Hz in prepare_input and stored
        # directly in _raw_desire.  When subsampling for the policy model,
        # .max() over each 2-step window merges pulses, matching the
        # InputQueues .max(axis=2) semantics in openpilot/modeld.
        self.prev_desire_vec = np.zeros(DESIRE_LEN, dtype=np.float32)

        # Traffic convention: right-hand traffic [left=0, right=1]
        self.traffic_convention = np.array([[1, 0]], dtype=np.float16)

        # Warp matrices — computed lazily on first prepare_input based on actual
        # rendered image dimensions (environment may render at different resolution
        # than requested in get_camera_configs).
        self.warp_road = None
        self.warp_wide = None
        self._warps_initialized = False

        # Cache for parse_output and run_inference
        self._cur_ego_state = None
        self._last_vision_out = None

    # ------------------------------------------------------------------
    # BaseModelAdapter interface
    # ------------------------------------------------------------------

    def load_model(self):
        """Load both ONNX sessions and metadata pickles."""
        print("Loading OpenPilot ONNX models...")

        vision_path = self.ckpt_dir / "driving_vision.onnx"
        policy_path = self.ckpt_dir / "driving_policy.onnx"
        vision_meta_path = self.ckpt_dir / "driving_vision_metadata.pkl"
        policy_meta_path = self.ckpt_dir / "driving_policy_metadata.pkl"

        for p in [vision_path, policy_path, vision_meta_path, policy_meta_path]:
            if not p.exists():
                raise FileNotFoundError(f"Required file not found: {p}")

        # Load metadata — slices are nested under 'output_slices' key
        with open(vision_meta_path, "rb") as f:
            vision_meta_raw = pickle.load(f)
        with open(policy_meta_path, "rb") as f:
            policy_meta_raw = pickle.load(f)

        self.vision_meta = vision_meta_raw.get("output_slices", vision_meta_raw)
        self.policy_meta = policy_meta_raw.get("output_slices", policy_meta_raw)

        print(f"  Vision metadata keys: {list(self.vision_meta.keys())}")
        print(f"  Policy metadata keys: {list(self.policy_meta.keys())}")

        # ONNX sessions with CUDA
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        print(f"  Loading vision model: {vision_path}")
        self.vision_session = ort.InferenceSession(
            str(vision_path), sess_options=sess_opts, providers=providers
        )

        print(f"  Loading policy model: {policy_path}")
        self.policy_session = ort.InferenceSession(
            str(policy_path), sess_options=sess_opts, providers=providers
        )

        # Log actual providers
        print(f"  Vision providers: {self.vision_session.get_providers()}")
        print(f"  Policy providers: {self.policy_session.get_providers()}")

        print("OpenPilot models loaded successfully.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """Two cameras matching OpenPilot's real hardware."""
        return {
            "CAM_ROAD": {
                "x": 0.0, "y": 0.0, "z": 1.22,
                "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                "fov": 40, "width": 1928, "height": 1208,
            },
            "CAM_WIDE": {
                "x": 0.0, "y": 0.0, "z": 1.22,
                "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                "fov": 120, "width": 1928, "height": 1208,
            },
        }

    def get_waypoint_dt(self) -> float:
        return 0.5

    def get_trajectory_time_horizon(self) -> float:
        return 4.0

    def prepare_input(
        self,
        images: Dict[str, np.ndarray],
        ego_state: Dict[str, Any],
        scenario_data: Dict[str, Any],
        frame_id: int,
    ) -> Any:
        """
        Called every sim step (10Hz).  Warps camera images, runs the vision
        model, and buffers the resulting features and desire pulses at 10Hz.
        """
        # --- Desire Mapping (NAVSIM/BridgeSim -> OpenPilot) ---
        # BridgeSim command (0-indexed): 0=LEFT, 1=RIGHT, 2=STRAIGHT,
        #   3=LANEFOLLOW, 4=CHANGELANELEFT, 5=CHANGELANERIGHT
        # OP Desire enum: 0=none, 1=turnLeft, 2=turnRight,
        #   3=laneChangeLeft, 4=laneChangeRight, 5=keepLeft, 6=keepRight
        # From what I could find, the only signals sent by OpenPilot are lane
        # changes on blinker inputs. All other desires seem to be legacy/unused.

        command = ego_state.get('command', 3)
        desire_map = {0: 0, 1: 0, 2: 0, 3: 0, 4: 3, 5: 4}
        desire_index = desire_map.get(command, 0)
        # Build current desire one-hot (matching modeld lines 357-359)
        cur_desire = np.zeros(DESIRE_LEN, dtype=np.float32)
        if 0 < desire_index < DESIRE_LEN:
            cur_desire[desire_index] = 1.0
        # Zero slot 0 ("none") so it never triggers a rising edge,
        # matching modeld line 191: inputs['desire_pulse'][0] = 0
        cur_desire[0] = 0.0

        # Rising-edge detection per element (modeld line 192):
        #   new_desire = where(cur - prev > .99, cur, 0)
        # Store directly in raw buffer at 10Hz; the .max() subsampling
        # in run_inference merges pulses into 5Hz context slots, matching
        # InputQueues .max(axis=2) semantics in openpilot/modeld.
        new_desire = np.where(cur_desire - self.prev_desire_vec > 0.99,
                              cur_desire, 0.0)
        self.prev_desire_vec[:] = cur_desire

        self._raw_desire[0, :-1] = self._raw_desire[0, 1:]
        self._raw_desire[0, -1] = new_desire.astype(np.float16)

        road_img = images["CAM_ROAD"][:, :, ::-1].copy()  # BGR→RGB
        wide_img = images["CAM_WIDE"][:, :, ::-1].copy()

        # Lazily compute warp matrices from actual rendered image dimensions
        if not self._warps_initialized:
            h, w = road_img.shape[:2]
            cam_configs = self.get_camera_configs()
            calib_euler = np.array([0.0, 0.0, 0.0])
            road_intr = _camera_intrinsics(
                fov_deg=cam_configs["CAM_ROAD"]["fov"], width=w, height=h
            )
            wide_intr = _camera_intrinsics(
                fov_deg=cam_configs["CAM_WIDE"]["fov"], width=w, height=h
            )
            self.warp_road = get_warp_matrix(calib_euler, road_intr, bigmodel_frame=False)
            self.warp_wide = get_warp_matrix(calib_euler, wide_intr, bigmodel_frame=True)
            self._warps_initialized = True
            print(f"  Warp matrices computed for actual image size {w}x{h}")

        # Perspective warp → model space (512×256)
        # Warp matrix maps model (dst) → camera (src), so use WARP_INVERSE_MAP
        # Use INTER_NEAREST to match OpenPilot's TinyGrad warp (Tensor.round)
        road_warped = cv2.warpPerspective(
            road_img, self.warp_road, MEDMODEL_INPUT_SIZE,
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        )
        wide_warped = cv2.warpPerspective(
            wide_img, self.warp_wide, MEDMODEL_INPUT_SIZE,
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        )


        road_yuv = rgb_to_yuv6ch(road_warped)  # (6, 128, 256)
        wide_yuv = rgb_to_yuv6ch(wide_warped)

        # Buffer frames
        self.road_frames.append(road_yuv)
        self.wide_frames.append(wide_yuv)

        # Build 2-frame input: frame[t-2] and frame[t], concatenated → (12, 128, 256)
        # prepare_input is called every sim step (10Hz), so 2 frames back = 200ms,
        # matching the model's expected temporal gap between input frames.
        if len(self.road_frames) >= 3:
            road_input = np.concatenate([self.road_frames[-3], self.road_frames[-1]], axis=0)
            wide_input = np.concatenate([self.wide_frames[-3], self.wide_frames[-1]], axis=0)
        else:
            # Not enough history — duplicate current frame
            road_input = np.concatenate([road_yuv, road_yuv], axis=0)
            wide_input = np.concatenate([wide_yuv, wide_yuv], axis=0)

        # Add batch dim → (1, 12, 128, 256)
        img = road_input[np.newaxis].astype(np.uint8)
        big_img = wide_input[np.newaxis].astype(np.uint8)

        # --- Run vision model at 10Hz ---
        vision_out = self.vision_session.run(
            None, {"img": img, "big_img": big_img}
        )[0]  # (1, 1576) float16

        # Extract hidden state and buffer at 10Hz
        hs_slice = self.vision_meta["hidden_state"]  # slice(1064, 1576)
        hidden_state = vision_out[0, hs_slice].astype(np.float16)  # (512,)

        self._raw_features[0, :-1] = self._raw_features[0, 1:]
        self._raw_features[0, -1] = hidden_state

        self._cur_ego_state = ego_state
        self._last_vision_out = vision_out

        return {"ego_state": ego_state}

    def run_inference(self, model_input: Any) -> Any:
        """
        Policy-only inference.  Vision already ran in prepare_input (10Hz)
        and buffered features.  Here we subsample to 5Hz and run the policy.
        """
        # --- Subsample raw buffers from 10Hz to 5Hz for policy model ---
        # Mirrors OpenPilot's InputQueues.get() logic:
        #   features: pick every 2nd step  (non-pulse key subsampling)
        #   desire:   .max() over 2-step windows  (pulse key merging)
        idxs = np.arange(-1, -self._raw_buffer_len - 1,
                         -self._steps_per_context)[::-1]
        features_for_policy = self._raw_features[:, idxs]
        desire_for_policy = self._raw_desire.reshape(
            1, 25, self._steps_per_context, DESIRE_LEN
        ).max(axis=2)

        # --- Policy model ---
        policy_out = self.policy_session.run(
            None,
            {
                "features_buffer": features_for_policy,
                "desire_pulse": desire_for_policy,
                "traffic_convention": self.traffic_convention,
            },
        )[0]  # (1, 1000) float16

        policy_out = policy_out.astype(np.float32)

        # --- Parse policy outputs ---
        plan_slice = self.policy_meta["plan"]             # slice(0, 990)
        desire_slice = self.policy_meta["desire_state"]   # slice(990, 998)

        raw_plan = policy_out[:, plan_slice]    # (1, 990)
        raw_desire = policy_out[:, desire_slice]  # (1, 8)

        # Check MHP format: if size == 2 * n_values, it's mu+std (no hypotheses)
        plan_n_values = IDX_N * PLAN_WIDTH  # 33 * 15 = 495
        if raw_plan.shape[1] == 2 * plan_n_values:
            # Not MHP: raw is [mu(495), std(495)]
            plan_in_N, plan_out_N = 0, 0
        else:
            # MHP: 5 hypotheses
            plan_in_N, plan_out_N = PLAN_MHP_N, 1
        plan = parse_mdn(raw_plan, in_N=plan_in_N, out_N=plan_out_N, out_shape=(IDX_N, PLAN_WIDTH))
        # plan shape: (1, 33, 15)

        desire_state = softmax(raw_desire, axis=-1)

        return {
            "plan": plan,
        }

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Extract trajectory from plan output and convert to BridgeSim ego frame.

        OpenPilot device frame: x=forward, y=right, z=down
        BridgeSim ego frame:    x=left, y=forward
        """
        plan = model_output["plan"][0]  # (33, 15)

        plan_x_fwd = plan[:, 0]
        plan_y_right = plan[:, 1]

        # Interpolate from non-uniform T_IDXS to uniform 0.5 s spacing,
        target_times = [t for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]]
        interp_x_fwd = np.interp(target_times, T_IDXS, plan_x_fwd)
        interp_y_right = np.interp(target_times, T_IDXS, plan_y_right)

        trajectory = np.column_stack([-interp_y_right, interp_x_fwd])  # (8, 2)

        return {"trajectory": trajectory}
