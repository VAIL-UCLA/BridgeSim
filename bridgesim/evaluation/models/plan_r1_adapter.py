"""
Model adapter for Plan-R1 (token-based autoregressive planner, NuPlan-trained).

Plan-R1 works by:
  1. Encoding agent history and map as a heterogeneous graph (torch_geometric HeteroData).
  2. Autoregressively decoding 16 action tokens (each token = 0.5s displacement).
  3. Returning predicted world-frame trajectories for all agents.

Mode options:
  - 'plan': uses plan_inference() — ego uses RL-fine-tuned plan_backbone, others use pred_backbone.
  - 'pred': uses pred_inference() — all agents use pred_backbone (IL baseline).

Vendored model code lives in bridgesim/modelzoo/plan_r1/.
"""

import math
import numpy as np
import torch
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Importing from bridgesim.modelzoo.plan_r1 triggers __init__.py which
# inserts the vendor directory into sys.path so all Plan-R1 bare imports resolve.
import bridgesim.modelzoo.plan_r1  # noqa: F401 — side-effect: sys.path setup

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PLAN_R1_ROOT = Path(__file__).resolve().parents[2] / "modelzoo" / "plan_r1"
_DEFAULT_TOKEN_DICT_PATH = str(_PLAN_R1_ROOT / "tokens" / "tokens_1024.pt")

# ---------------------------------------------------------------------------
# Default model hyper-parameters (match released checkpoint)
# ---------------------------------------------------------------------------
_DEFAULT_NUM_TOKENS = 1024
_DEFAULT_INTERVAL = 5           # subsample factor: 10Hz → 2Hz tokens
_DEFAULT_NUM_HIST = 20          # 2s history @ 10Hz
_DEFAULT_NUM_FUTURE = 80        # 8s future @ 10Hz
_DEFAULT_HIDDEN_DIM = 128
_DEFAULT_NUM_HEADS = 8
_DEFAULT_NUM_ATTN_LAYERS = 6
_DEFAULT_AGENT_RADIUS = 60.0
_DEFAULT_POLYGON_RADIUS = 30.0
_DEFAULT_NUM_HOPS = 4
_DEFAULT_MAX_AGENTS = 60
_DEFAULT_MAP_RADIUS = 120.0     # search radius for map features (metres)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class PlanR1Adapter(BaseModelAdapter):
    """
    Adapter for Plan-R1 — a vector-input (no camera) token-based autoregressive planner.
    """

    def __init__(
        self,
        checkpoint_path: str,
        token_dict_path: Optional[str] = None,
        mode: str = "plan",
        **kwargs,
    ):
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self._token_dict_path = token_dict_path or _DEFAULT_TOKEN_DICT_PATH
        self._mode = mode  # 'plan' or 'pred'

        # These are set in load_model() from the loaded checkpoint's hparams
        self._interval: int = _DEFAULT_INTERVAL
        self._num_hist: int = _DEFAULT_NUM_HIST
        self._num_hist_intervals: int = _DEFAULT_NUM_HIST // _DEFAULT_INTERVAL  # 4

        # Per-frame state saved during prepare_input for use in parse_output
        self._center_position: Optional[np.ndarray] = None
        self._center_heading: float = 0.0

        # Temporal history: deque of per-frame dicts keyed by agent id
        # Each entry: {'position': np.ndarray [2], 'heading': float,
        #              'velocity': np.ndarray [2], 'type': int,
        #              'length': float, 'width': float}
        _maxlen = _DEFAULT_NUM_HIST + 1  # 21 frames
        self._ego_history: deque = deque(maxlen=_maxlen)
        self._neighbor_history: Dict[str, deque] = {}
        self._neighbor_maxlen: int = _maxlen

        # TokenBuilder instance (created in load_model after token dict path resolved)
        self._token_builder = None

    # ------------------------------------------------------------------ #
    # BaseModelAdapter interface                                           #
    # ------------------------------------------------------------------ #

    def load_model(self):
        from model.PlanR1 import PlanR1  # bare import resolved via sys.path
        from transforms.token_builder import TokenBuilder

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        # Support both raw state_dict checkpoints and PyTorch-Lightning checkpoints
        if "hyper_parameters" in ckpt:
            hparams = ckpt["hyper_parameters"]
            # Ensure token_dict_path matches vendored copy
            hparams["token_dict_path"] = self._token_dict_path
            self.model = PlanR1(**hparams)
            state_dict = ckpt.get("state_dict", {})
            # Strip module. prefix (DDP)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
        else:
            # Attempt load_from_checkpoint (Lightning)
            self.model = PlanR1.load_from_checkpoint(
                self.checkpoint_path,
                map_location="cpu",
                strict=False,
                token_dict_path=self._token_dict_path,
                mode=self._mode,
            )

        self.model.to(self.device)
        self.model.eval()

        # Cache hparams for prepare_input
        self._interval = self.model.interval
        self._num_hist = self.model.num_historical_steps
        self._num_hist_intervals = self.model.num_historical_intervals
        self._neighbor_maxlen = self._num_hist + 1

        # Token builder (runs on CPU tensors, moved to device later)
        self._token_builder = TokenBuilder(
            token_dict_path=self._token_dict_path,
            interval=self._interval,
            num_historical_steps=self._num_hist,
            mode="pred",
        )
        print(f"[PlanR1Adapter] Loaded {self._mode} model from {self.checkpoint_path}")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        # Plan-R1 is vector-only; provide a dummy front camera for visualisation
        return {
            "CAM_F0": {
                "fov": 70.0, "x": 1.5, "y": 0.0, "z": 1.5,
                "yaw": 0.0, "pitch": -5.0, "roll": 0.0,
            }
        }

    def get_waypoint_dt(self) -> float:
        return 0.5  # 16 tokens × 0.5s = 8s horizon

    def get_trajectory_time_horizon(self) -> float:
        return 8.0

    def prepare_input(
        self,
        images: Dict[str, np.ndarray],
        ego_state: Dict[str, Any],
        scenario_data: Dict[str, Any],
        frame_id: int,
    ) -> Any:
        """Build a torch_geometric HeteroData Batch from MetaBench scenario data."""
        from torch_geometric.data import HeteroData, Batch

        # ---- 1. Update ego history ----------------------------------------
        ego_x = float(ego_state["position"][0])
        ego_y = float(ego_state["position"][1])
        ego_heading = float(ego_state["heading"])
        ego_vel = ego_state.get("velocity", [0.0, 0.0, 0.0])
        self._ego_history.append({
            "position": np.array([ego_x, ego_y], dtype=np.float64),
            "heading": ego_heading,
            "velocity": np.array([float(ego_vel[0]), float(ego_vel[1])], dtype=np.float32),
        })
        self._center_position = np.array([ego_x, ego_y], dtype=np.float64)
        self._center_heading = ego_heading

        # ---- 2. Update neighbor history ------------------------------------
        tracks = scenario_data.get("tracks", {})
        sdc_id = scenario_data.get("metadata", {}).get("sdc_id", None)
        for obj_id, track in tracks.items():
            if obj_id == sdc_id:
                continue
            positions = track["state"]["position"]
            valid = track["state"].get("valid", None)
            if frame_id >= len(positions):
                continue
            if valid is not None and not valid[frame_id]:
                continue
            pos = positions[frame_id][:2]
            heading = float(track["state"]["heading"][frame_id])
            vel_arr = track["state"].get("velocity", None)
            vel = np.array(vel_arr[frame_id][:2], dtype=np.float32) if vel_arr is not None else np.zeros(2, np.float32)

            if obj_id not in self._neighbor_history:
                self._neighbor_history[obj_id] = deque(maxlen=self._neighbor_maxlen)
            self._neighbor_history[obj_id].append({
                "position": np.array(pos[:2], dtype=np.float64),
                "heading": heading,
                "velocity": vel,
                "frame_id": frame_id,
                "type": self._get_agent_type(track.get("type", "VEHICLE")),
                "length": float(np.squeeze(track["state"].get("length", np.array([4.5]))).flat[0]) if track["state"].get("length") is not None else 4.5,
                "width": float(np.squeeze(track["state"].get("width", np.array([2.0]))).flat[0]) if track["state"].get("width") is not None else 2.0,
            })

        # ---- 3. Collect agents (ego + neighbors sorted by distance) --------
        T = self._num_hist + 1  # 21
        center_pos = self._center_position  # shape [2]

        # Ego box: [front_half, rear_half, left_half, right_half]
        # For ego we use a default car size (4.8m × 2.0m centred)
        ego_length = float(ego_state.get("length", 4.8)) if isinstance(ego_state.get("length"), (int, float)) else 4.8
        ego_width = float(ego_state.get("width", 2.0)) if isinstance(ego_state.get("width"), (int, float)) else 2.0

        # Gather neighbor agents at current frame, sort by distance
        neighbor_infos: List[Tuple[str, float]] = []
        for obj_id, hist in self._neighbor_history.items():
            if len(hist) == 0:
                continue
            last = hist[-1]
            if last["frame_id"] != frame_id:
                continue
            dist = float(np.linalg.norm(last["position"] - center_pos))
            if dist > _DEFAULT_MAX_AGENTS * 2:  # crude pre-filter
                continue
            neighbor_infos.append((obj_id, dist))
        neighbor_infos.sort(key=lambda x: x[1])
        neighbor_infos = neighbor_infos[:_DEFAULT_MAX_AGENTS]

        N = 1 + len(neighbor_infos)  # ego + neighbours
        agent_position = np.zeros((N, T, 2), dtype=np.float64)
        agent_heading = np.zeros((N, T), dtype=np.float32)
        agent_velocity = np.zeros((N, T, 2), dtype=np.float32)
        agent_visible_mask = np.zeros((N, T), dtype=bool)
        agent_type = np.zeros(N, dtype=np.uint8)     # 0=Vehicle
        agent_identity = np.zeros(N, dtype=np.uint8)  # 0=Ego, 1=Agent
        agent_box = np.zeros((N, 4), dtype=np.float32)

        # Ego (index 0)
        agent_identity[0] = 0  # Ego
        agent_box[0] = [ego_length / 2, ego_length / 2, ego_width / 2, ego_width / 2]
        for t_idx, state in enumerate(self._ego_history):
            # Map deque position to T-slot: last element → slot T-1
            slot = T - len(self._ego_history) + t_idx
            if 0 <= slot < T:
                agent_position[0, slot] = state["position"]
                agent_heading[0, slot] = state["heading"]
                agent_velocity[0, slot] = state["velocity"]
                agent_visible_mask[0, slot] = True

        # Neighbours
        for agent_idx, (obj_id, _) in enumerate(neighbor_infos, start=1):
            hist = self._neighbor_history[obj_id]
            last = hist[-1]
            agent_type[agent_idx] = last["type"]
            agent_identity[agent_idx] = 1  # Agent
            hl = last["length"]
            hw = last["width"]
            agent_box[agent_idx] = [hl / 2, hl / 2, hw / 2, hw / 2]
            for t_idx, state in enumerate(hist):
                slot = T - len(hist) + t_idx
                if 0 <= slot < T:
                    agent_position[agent_idx, slot] = state["position"]
                    agent_heading[agent_idx, slot] = state["heading"]
                    agent_velocity[agent_idx, slot] = state["velocity"]
                    agent_visible_mask[agent_idx, slot] = True

        # Centre all positions on current ego position (world-relative-to-ego)
        agent_position = np.where(
            agent_visible_mask[:, :, np.newaxis],
            agent_position - center_pos[np.newaxis, np.newaxis, :],
            0.0,
        )

        # Convert to torch tensors
        t_agent_position = torch.tensor(agent_position, dtype=torch.float32)
        t_agent_heading = torch.tensor(agent_heading, dtype=torch.float32)
        t_agent_velocity = torch.tensor(agent_velocity, dtype=torch.float32)
        t_agent_visible_mask = torch.tensor(agent_visible_mask, dtype=torch.bool)
        t_agent_type = torch.tensor(agent_type, dtype=torch.uint8)
        t_agent_identity = torch.tensor(agent_identity, dtype=torch.uint8)
        t_agent_box = torch.tensor(agent_box, dtype=torch.float32)

        # ---- 4. Token-quantise historical trajectory ----------------------
        # Build a minimal HeteroData that TokenBuilder expects
        data = HeteroData()
        data["agent"]["num_nodes"] = N
        data["agent"]["type"] = t_agent_type
        data["agent"]["identity"] = t_agent_identity
        data["agent"]["box"] = t_agent_box
        data["agent"]["position"] = t_agent_position          # [N, T, 2]
        data["agent"]["heading"] = t_agent_heading             # [N, T]
        data["agent"]["velocity"] = t_agent_velocity          # [N, T, 2]
        data["agent"]["visible_mask"] = t_agent_visible_mask  # [N, T]

        # TokenBuilder modifies data in-place and adds recon_* fields
        data = self._token_builder(data)

        # ---- 5. Initialise infer_* fields ---------------------------------
        H = self._num_hist_intervals  # 4
        data["agent"]["infer_position"] = data["agent"]["recon_position"][:, :H].clone()
        data["agent"]["infer_heading"] = data["agent"]["recon_heading"][:, :H].clone()
        data["agent"]["infer_token"] = data["agent"]["recon_token"][:, :H].clone()
        data["agent"]["infer_token_mask"] = data["agent"]["recon_token_mask"][:, :H].clone()
        data["agent"]["infer_valid_mask"] = data["agent"]["recon_valid_mask"][:, :H].clone()
        # batch index for unbatch operations
        data["agent"]["batch"] = torch.zeros(N, dtype=torch.long)

        # ---- 6. Build polygon + polyline map graph -------------------------
        data = self._build_map_graph(data, scenario_data, frame_id)

        # ---- 7. Wrap in Batch and move to device --------------------------
        batch = Batch.from_data_list([data])
        batch = batch.to(self.device)
        return batch

    def run_inference(self, model_input: Any) -> Any:
        with torch.no_grad():
            if self._mode == "plan":
                data, position, heading, valid_mask = self.model.plan_inference(model_input)
            else:
                data, position, heading, valid_mask = self.model.pred_inference(model_input)
        return (position, heading, valid_mask)

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        position, heading, valid_mask = model_output
        # position: [N_total, num_future_intervals, 2]  — world-relative-to-center_position
        # Ego is always agent index 0
        ego_traj = position[0].cpu().numpy()   # [16, 2]

        # Convert from world-relative-to-ego to ego-local frame.
        # The stored positions are (world - center_position), still in world orientation.
        # Rotate by -center_heading to get ego-local frame.
        cos_h = np.cos(self._center_heading)
        sin_h = np.sin(self._center_heading)
        local_x = cos_h * ego_traj[:, 0] + sin_h * ego_traj[:, 1]  # forward
        local_y = -sin_h * ego_traj[:, 0] + cos_h * ego_traj[:, 1]  # lateral

        # MetaBench expects (N, 2) with columns (lateral, forward)
        trajectory = np.stack([local_y, local_x], axis=-1)  # [16, 2]
        return {"trajectory": trajectory}

    def reset_temporal_history(self):
        """Clear history buffers between scenarios."""
        self._ego_history.clear()
        self._neighbor_history.clear()

    # ------------------------------------------------------------------ #
    # Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_agent_type(obj_type: Any) -> int:
        """Map MetaBench agent type string to Plan-R1 type index (0=Vehicle, 1=Ped, 2=Bicycle)."""
        if not isinstance(obj_type, str):
            return 0
        t = obj_type.lower()
        if "pedestrian" in t:
            return 1
        if "bicycle" in t or "cyclist" in t:
            return 2
        return 0

    def _build_map_graph(
        self,
        data,
        scenario_data: Dict[str, Any],
        frame_id: int,
    ):
        """
        Build polygon + polyline graph nodes and edges from scenario_data['map_features'].

        Plan-R1 MapEncoder expects:
          data['polygon']: position, heading, heading_valid_mask, type, speed_limit,
                           speed_limit_valid_mask, traffic_light, on_route_mask, num_nodes, batch
          data['polyline']: position, heading, length, batch
          data['polyline','polygon']['polyline_to_polygon_edge_index']: [2, E_l2g]
          data['polygon','polygon']['left_edge_index']: [2, E_g2g]  (may be empty)
          data['polygon','polygon']['right_edge_index']
          data['polygon','polygon']['incoming_edge_index']
          data['polygon','polygon']['outgoing_edge_index']
        """
        # _polygon_types: LANE=0, CROSSWALK=1, DRIVABLE_AREA_SEGMENT=2, STATIC_OBJECT=3
        # _traffic_light_types: GREEN=0, YELLOW=1, RED=2, UNKNOWN=3, NONE=4
        map_features = scenario_data.get("map_features", {})
        center_pos = self._center_position  # np [2] float64

        polygon_positions: List[List[float]] = []
        polygon_headings: List[float] = []
        polygon_heading_valid: List[bool] = []
        polygon_types: List[int] = []
        polygon_traffic_light: List[int] = []  # default NONE=4
        polygon_on_route: List[bool] = []
        polygon_speed_limit: List[float] = []
        polygon_speed_limit_valid: List[bool] = []

        polyline_positions: List[List[float]] = []
        polyline_headings: List[float] = []
        polyline_lengths: List[float] = []
        # For each polyline, which polygon does it belong to?
        l2g_src: List[int] = []  # polyline index
        l2g_dst: List[int] = []  # polygon index

        # --- Determine on-route roadblock IDs from metadata -----------------
        # metadata['roadblock_ids_per_frame'][frame_id] lists the roadblock IDs
        # the ego occupies at each frame.  map_features[lane]['roadblock_id'] is
        # the roadblock a lane belongs to.  Matching gives exact on-route labels.
        roadblock_ids_per_frame = scenario_data.get("metadata", {}).get(
            "roadblock_ids_per_frame", None
        )
        if roadblock_ids_per_frame is None:
            raise ValueError(
                "scenario_data['metadata']['roadblock_ids_per_frame'] is missing. "
                "Plan-R1 requires this field to determine on-route lanes."
            )
        if frame_id >= len(roadblock_ids_per_frame):
            raise ValueError(
                f"frame_id={frame_id} out of range for roadblock_ids_per_frame "
                f"(len={len(roadblock_ids_per_frame)})"
            )
        route_roadblock_ids = set(str(rid) for rid in roadblock_ids_per_frame[frame_id])

        # --- Pass 1: collect all valid features and assign polygon indices ---
        # Use str(feat_id) as canonical key so topology lookups (which store
        # neighbour IDs as strings) can match regardless of original key type.
        feat_id_to_poly_idx: Dict[str, int] = {}
        # Also keep original key so we can look up map_features[original_key]
        str_to_orig_key: Dict[str, Any] = {}

        for feat_id, feat in map_features.items():
            feat_type = feat.get("type", "")
            if not isinstance(feat_type, str):
                continue
            polyline_raw = feat.get("polyline", None)
            if polyline_raw is None:
                continue
            polyline_raw = np.array(polyline_raw, dtype=np.float64)
            if polyline_raw.ndim < 2 or len(polyline_raw) < 2:
                continue
            pts_xy = polyline_raw[:, :2]
            dists = np.linalg.norm(pts_xy - center_pos[np.newaxis, :], axis=-1)
            if dists.min() > _DEFAULT_MAP_RADIUS:
                continue
            ft_upper = feat_type.upper()
            if "LANE" not in ft_upper and "CROSSWALK" not in ft_upper:
                continue
            sid = str(feat_id)
            feat_id_to_poly_idx[sid] = len(feat_id_to_poly_idx)
            str_to_orig_key[sid] = feat_id

        # --- Pass 2: build node features and polyline nodes ----------------
        ordered_feat_ids = list(feat_id_to_poly_idx.keys())  # str keys, insertion order

        for feat_id in ordered_feat_ids:
            feat = map_features[str_to_orig_key[feat_id]]
            feat_type = feat.get("type", "").upper()
            poly_type = 0 if "LANE" in feat_type else 1  # LANE=0, CROSSWALK=1

            pts_centred = np.array(feat["polyline"], dtype=np.float64)[:, :2] - center_pos[np.newaxis, :]
            diffs = pts_centred[1:] - pts_centred[:-1]
            seg_lengths = np.linalg.norm(diffs, axis=-1)
            seg_headings = np.arctan2(diffs[:, 1], diffs[:, 0])

            mid_idx = len(diffs) // 2
            poly_pos = pts_centred[mid_idx].tolist()
            poly_heading = float(seg_headings[mid_idx])

            # on_route: lane's roadblock_id is in the current frame's route set
            lane_rb_id = feat.get("roadblock_id", None)
            on_route = (lane_rb_id is not None) and (str(lane_rb_id) in route_roadblock_ids)

            poly_idx = feat_id_to_poly_idx[feat_id]
            polygon_positions.append(poly_pos)
            polygon_headings.append(poly_heading)
            polygon_heading_valid.append(True)
            polygon_types.append(poly_type)
            polygon_traffic_light.append(4)  # NONE by default
            polygon_on_route.append(on_route)
            polygon_speed_limit.append(0.0)
            polygon_speed_limit_valid.append(False)

            # Polyline segments → nodes
            for seg_i in range(len(diffs)):
                if seg_lengths[seg_i] < 1e-6:
                    continue
                polyline_positions.append(pts_centred[seg_i].tolist())
                polyline_headings.append(float(seg_headings[seg_i]))
                polyline_lengths.append(float(seg_lengths[seg_i]))
                l2g_src.append(len(polyline_positions) - 1)
                l2g_dst.append(poly_idx)

        # --- Pass 3: build topology edge indices from map_features fields --
        # map_features stores neighbour IDs as strings; we only keep edges
        # where both endpoints were included (passed proximity filter above).
        left_src, left_dst = [], []
        right_src, right_dst = [], []
        incoming_src, incoming_dst = [], []
        outgoing_src, outgoing_dst = [], []

        for feat_id in ordered_feat_ids:
            feat = map_features[str_to_orig_key[feat_id]]
            src_idx = feat_id_to_poly_idx[feat_id]

            left_nb = feat.get("left_neighbor") or []
            if isinstance(left_nb, str):
                left_nb = [left_nb]
            for nb_id in left_nb:
                if str(nb_id) in feat_id_to_poly_idx:
                    left_src.append(src_idx)
                    left_dst.append(feat_id_to_poly_idx[str(nb_id)])

            right_nb = feat.get("right_neighbor") or []
            if isinstance(right_nb, str):
                right_nb = [right_nb]
            for nb_id in right_nb:
                if str(nb_id) in feat_id_to_poly_idx:
                    right_src.append(src_idx)
                    right_dst.append(feat_id_to_poly_idx[str(nb_id)])

            for nb_id in (feat.get("entry_lanes") or []):
                if str(nb_id) in feat_id_to_poly_idx:
                    incoming_src.append(feat_id_to_poly_idx[str(nb_id)])
                    incoming_dst.append(src_idx)

            for nb_id in (feat.get("exit_lanes") or []):
                if str(nb_id) in feat_id_to_poly_idx:
                    outgoing_src.append(src_idx)
                    outgoing_dst.append(feat_id_to_poly_idx[str(nb_id)])

        def _make_edge_index(src, dst):
            if src:
                return torch.tensor([src, dst], dtype=torch.long)
            return torch.zeros(2, 0, dtype=torch.long)

        M = len(polygon_positions)
        L = len(polyline_positions)

        if M == 0:
            polygon_positions = [[0.0, 0.0]]
            polygon_headings = [0.0]
            polygon_heading_valid = [False]
            polygon_types = [0]
            polygon_traffic_light = [4]
            polygon_on_route = [False]
            polygon_speed_limit = [0.0]
            polygon_speed_limit_valid = [False]
            M = 1

        if L == 0:
            polyline_positions = [[0.0, 0.0]]
            polyline_headings = [0.0]
            polyline_lengths = [1.0]
            l2g_src = [0]
            l2g_dst = [0]
            L = 1

        data["polygon"]["num_nodes"] = M
        data["polygon"]["position"] = torch.tensor(polygon_positions, dtype=torch.float32)
        data["polygon"]["heading"] = torch.tensor(polygon_headings, dtype=torch.float32)
        data["polygon"]["heading_valid_mask"] = torch.tensor(polygon_heading_valid, dtype=torch.bool)
        data["polygon"]["type"] = torch.tensor(polygon_types, dtype=torch.uint8)
        data["polygon"]["traffic_light"] = torch.tensor(polygon_traffic_light, dtype=torch.uint8)
        data["polygon"]["on_route_mask"] = torch.tensor(polygon_on_route, dtype=torch.bool)
        data["polygon"]["speed_limit"] = torch.tensor(polygon_speed_limit, dtype=torch.float32)
        data["polygon"]["speed_limit_valid_mask"] = torch.tensor(polygon_speed_limit_valid, dtype=torch.bool)
        data["polygon"]["batch"] = torch.zeros(M, dtype=torch.long)

        data["polyline"]["position"] = torch.tensor(polyline_positions, dtype=torch.float32)
        data["polyline"]["heading"] = torch.tensor(polyline_headings, dtype=torch.float32)
        data["polyline"]["length"] = torch.tensor(polyline_lengths, dtype=torch.float32)
        data["polyline"]["batch"] = torch.zeros(L, dtype=torch.long)

        data["polyline", "polygon"]["polyline_to_polygon_edge_index"] = torch.tensor(
            [l2g_src, l2g_dst], dtype=torch.long
        )
        data["polygon", "polygon"]["left_edge_index"] = _make_edge_index(left_src, left_dst)
        data["polygon", "polygon"]["right_edge_index"] = _make_edge_index(right_src, right_dst)
        data["polygon", "polygon"]["incoming_edge_index"] = _make_edge_index(incoming_src, incoming_dst)
        data["polygon", "polygon"]["outgoing_edge_index"] = _make_edge_index(outgoing_src, outgoing_dst)

        return data
