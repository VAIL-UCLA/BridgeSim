"""
Alpamayo-R1 adapter (BridgeSim -> Alpamayo via separate venv)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple
from collections import deque

import numpy as np

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter


def _bridgesim_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_alpamayo_root() -> Path:
    return _bridgesim_repo_root() / "bridgesim" / "modelzoo" / "nvidia" / "alpamayo"


def _default_alpamayo_python() -> str:
    return os.environ.get(
        "ALPAMAYO_PYTHON",
        str(_default_alpamayo_root() / "ar1_venv" / "bin" / "python"),
    )


def _default_alpamayo_script() -> str:
    return os.environ.get(
        "ALPAMAYO_SCRIPT",
        str(_bridgesim_repo_root() / "bridgesim" / "modelzoo" / "nvidia" / "tools" / "alpamayo_bridgesim_infer.py"),
    )


@dataclass
class AlpamayoR1AdapterConfig:
    # 4-camera setup (cam-major, 4 frames each -> 16 frames total)
    camera_names: Tuple[str, ...] = (
        "CAM_CROSS_LEFT",
        "CAM_FRONT_WIDE",
        "CAM_CROSS_RIGHT",
        "CAM_FRONT_TELE",
    )
    frames_per_camera: int = 4
    fallback_camera: str = "rgb_camera"

    # Alpamayo venv python + BridgeSim glue script
    alp_python: str = field(default_factory=_default_alpamayo_python)
    alp_script: str = field(default_factory=_default_alpamayo_script)

    # Alpamayo generation params
    top_p: float = 0.98
    temperature: float = 0.6
    num_traj_samples: int = 1
    max_generation_length: int = 256
    coord_mode: str = "x_forward_y_left"
    max_history: int = 20  # ego history length (world frame)

    # Trajectory metadata
    waypoint_dt: float = 0.1
    time_horizon_s: float = 6.4  # Alpamayo outputs ~64 steps @ 10Hz

    # Debug
    cuda_launch_blocking: bool = False


class AlpamayoSubprocessClient:
    """
    Calls Alpamayo inference using a dedicated venv python.
    """

    def __init__(self, alp_python: str, alp_script: str, model_name: str):
        self.alp_python = alp_python
        self.alp_script = alp_script
        self.model_name = model_name

    def infer(
        self,
        image_stack_uint8_nhwc: np.ndarray,
        ego_xyz_world: np.ndarray,
        ego_rot_yaw: np.ndarray,
        *,
        top_p: float,
        temperature: float,
        num_traj_samples: int,
        max_generation_length: int,
        nav_cmd: str,
        coord_mode: str,
        cuda_launch_blocking: bool,
    ) -> dict:
        """
        Run one Alpamayo inference in a separate venv subprocess.
        """
        img = image_stack_uint8_nhwc
        if img.ndim == 3:
            img = img[None, ...]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            img_npy = td / "img.npy"
            ego_xyz_npy = td / "ego_xyz.npy"
            ego_rot_npy = td / "ego_rot.npy"
            out_json = td / "out.json"

            np.save(img_npy, img)
            np.save(ego_xyz_npy, ego_xyz_world)
            np.save(ego_rot_npy, ego_rot_yaw)

            cmd = [
                "env",
                "-u",
                "PYTHONUTF8",
                "-u",
                "PYTHONHOME",
                "-u",
                "PYTHONPATH",
            ]

            if cuda_launch_blocking:
                cmd += ["CUDA_LAUNCH_BLOCKING=1"]

            cmd += [
                str(self.alp_python),
                str(self.alp_script),
                "--model",
                str(self.model_name),
                "--image-npy",
                str(img_npy),
                "--ego-xyz-npy",
                str(ego_xyz_npy),
                "--ego-rot-npy",
                str(ego_rot_npy),
                "--out-json",
                str(out_json),
                "--top-p",
                str(top_p),
                "--temperature",
                str(temperature),
                "--num-traj-samples",
                str(num_traj_samples),
                "--max-generation-length",
                str(max_generation_length),
                "--nav-cmd",
                str(nav_cmd),
                "--coord-mode",
                str(coord_mode),
            ]

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if proc.returncode != 0:
                raise RuntimeError(
                    f"[AlpamayoSubprocessClient] Alpamayo subprocess failed (code={proc.returncode})\n"
                    f"CMD:\n{' '.join(cmd)}\n"
                    f"STDOUT:\n{proc.stdout}\n"
                    f"STDERR:\n{proc.stderr}\n"
                )

            if not out_json.exists():
                raise RuntimeError(
                    f"[AlpamayoSubprocessClient] Expected output JSON not found: {out_json}\n"
                    f"CMD:\n{' '.join(cmd)}\n"
                    f"STDOUT:\n{proc.stdout}\n"
                    f"STDERR:\n{proc.stderr}\n"
                )

            return json.loads(out_json.read_text())


class AlpamayoR1Adapter(BaseModelAdapter):
    """
    BridgeSim adapter for Alpamayo-R1, using a separate Alpamayo venv.
    """

    def __init__(self, checkpoint_path: str, config_path: str = None, **kwargs):
        super().__init__(checkpoint_path, config_path=config_path, **kwargs)

        # Allow CLI overrides; fallback to envvar/repo defaults if empty
        alp_python = kwargs.get("alp_python") or _default_alpamayo_python()
        alp_script = kwargs.get("alp_script") or _default_alpamayo_script()

        self.cfg = AlpamayoR1AdapterConfig(
            camera_names=tuple(kwargs.get("camera_names", AlpamayoR1AdapterConfig.camera_names)),
            frames_per_camera=int(kwargs.get("frames_per_camera", AlpamayoR1AdapterConfig.frames_per_camera)),
            fallback_camera=str(kwargs.get("fallback_camera", AlpamayoR1AdapterConfig.fallback_camera)),
            alp_python=str(alp_python),
            alp_script=str(alp_script),
            top_p=float(kwargs.get("top_p", 0.98)),
            temperature=float(kwargs.get("temperature", 0.6)),
            num_traj_samples=int(kwargs.get("num_traj_samples", 1)),
            max_generation_length=int(kwargs.get("max_generation_length", 256)),
            coord_mode=str(kwargs.get("coord_mode", "x_forward_y_left")),
            max_history=int(kwargs.get("max_history", 20)),
            waypoint_dt=float(kwargs.get("waypoint_dt", 0.1)),
            time_horizon_s=float(kwargs.get("time_horizon_s", 6.4)),
            cuda_launch_blocking=bool(kwargs.get("cuda_launch_blocking", False)),
        )

        # Buffer last K frames per camera
        self._img_hist_by_cam: Dict[str, Deque[np.ndarray]] = {
            cam: deque(maxlen=self.cfg.frames_per_camera) for cam in self.cfg.camera_names
        }

        # Ego pose history (WORLD frame; infer script converts to ego-local @ t0)
        self._ego_pos_hist: Deque[np.ndarray] = deque(maxlen=self.cfg.max_history)
        self._ego_yaw_hist: Deque[float] = deque(maxlen=self.cfg.max_history)

        self.client: Optional[AlpamayoSubprocessClient] = None

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """
        Request a 4-camera setup + a CAM_F0 alias (for visualization only).
        """
        cfg: Dict[str, Dict[str, float]] = {
            "CAM_FRONT_WIDE": {
                "x": 0.80,
                "y": 0.0,
                "z": 1.60,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "fov": 120,
                "width": 1920,
                "height": 1080,
            },
            "CAM_FRONT_TELE": {
                "x": 0.80,
                "y": 0.0,
                "z": 1.60,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "fov": 30,
                "width": 1920,
                "height": 1080,
            },
            "CAM_CROSS_LEFT": {
                "x": 0.40,
                "y": -0.55,
                "z": 1.60,
                "yaw": -90.0,
                "pitch": 0.0,
                "roll": 0.0,
                "fov": 120,
                "width": 1920,
                "height": 1080,
            },
            "CAM_CROSS_RIGHT": {
                "x": 0.40,
                "y": 0.55,
                "z": 1.60,
                "yaw": 90.0,
                "pitch": 0.0,
                "roll": 0.0,
                "fov": 120,
                "width": 1920,
                "height": 1080,
            },
        }

        # Alias: projection math in render_cam_f0_vis uses this
        cfg["CAM_F0"] = cfg["CAM_FRONT_WIDE"]
        return cfg

    def perceive(self, env, frame_id) -> Dict[str, np.ndarray]:
        """
        Capture only the 4 real Alpamayo cameras, and add CAM_F0 as an alias
        to CAM_FRONT_WIDE in the returned dict.
        """
        sensor = env.engine.get_sensor("rgb_camera")
        cam_cfgs_all = self.get_camera_configs()

        # Capture only real cams (exclude alias to prevent double-render)
        cam_cfgs_capture = {k: v for k, v in cam_cfgs_all.items() if k != "CAM_F0"}

        imgs: Dict[str, np.ndarray] = {}
        for name, cam_cfg in cam_cfgs_capture.items():
            sensor.lens.setFov(cam_cfg["fov"])

            sensor_output = sensor.perceive(
                to_float=False,
                new_parent_node=env.agent.origin,
                position=(cam_cfg["y"], cam_cfg["x"], cam_cfg["z"]),
                hpr=(-cam_cfg["yaw"], cam_cfg["pitch"], -cam_cfg["roll"]),
            )

            sensor_data = sensor_output.get() if hasattr(sensor_output, "get") else sensor_output
            imgs[name] = sensor_data

        # Add CAM_F0 alias (no extra rendering)
        if "CAM_FRONT_WIDE" in imgs:
            imgs["CAM_F0"] = imgs["CAM_FRONT_WIDE"]  # or imgs["CAM_FRONT_WIDE"].copy() if needed for isolation

        return imgs

    def load_model(self):
        """
        Validate alpamayo python/script exist and initialize the subprocess client.
        Alpamayo weights are loaded inside the subprocess.
        """
        print("Loading Alpamayo-R1 adapter (venv subprocess mode)...")
        print(f"  alp_python: {self.cfg.alp_python}")
        print(f"  alp_script: {self.cfg.alp_script}")

        if not os.path.isfile(self.cfg.alp_python):
            raise FileNotFoundError(
                f"Alpamayo venv python not found: {self.cfg.alp_python}\n"
                f"Set ALPAMAYO_PYTHON to your alpamayo venv python.\n"
                f"Example:\n  export ALPAMAYO_PYTHON=/path/to/alpamayo/ar1_venv/bin/python"
            )

        if not os.path.isfile(self.cfg.alp_script):
            raise FileNotFoundError(
                f"Alpamayo glue script not found: {self.cfg.alp_script}\n"
                f"Expected it in BridgeSim/tools/bridgesim_infer_once.py or set ALPAMAYO_SCRIPT."
            )

        self.client = AlpamayoSubprocessClient(
            alp_python=self.cfg.alp_python,
            alp_script=self.cfg.alp_script,
            model_name=self.checkpoint_path,  # HF repo id
        )
        print("Alpamayo-R1 adapter ready (model loads inside subprocess on first call).")

    def prepare_input(
        self,
        images: Dict[str, np.ndarray],
        ego_state: Dict[str, Any],
        scenario_data: Dict[str, Any],
        frame_id: int,
    ) -> Any:
        # Reset buffers at episode start
        if frame_id == 0:
            for dq in self._img_hist_by_cam.values():
                dq.clear()
            self._ego_pos_hist.clear()
            self._ego_yaw_hist.clear()

        # Push newest frame into each camera deque (only the 4 Alpamayo cams)
        for cam in self.cfg.camera_names:
            img = images.get(cam, None)
            if img is None:
                img = images.get(self.cfg.fallback_camera, None)
            if img is None and len(images) > 0:
                img = next(iter(images.values()))
            if img is None:
                img = np.zeros((320, 576, 3), dtype=np.uint8)

            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

            self._img_hist_by_cam[cam].append(img)

        # Flatten cam-major with left-padding per cam (4 cams × 4 frames = 16)
        frames_flat: list[np.ndarray] = []
        for cam in self.cfg.camera_names:
            dq = self._img_hist_by_cam[cam]
            if len(dq) == 0:
                pad = np.zeros((320, 576, 3), dtype=np.uint8)
                frames = [pad] * self.cfg.frames_per_camera
            else:
                frames = list(dq)
                if len(frames) < self.cfg.frames_per_camera:
                    frames = [frames[0]] * (self.cfg.frames_per_camera - len(frames)) + frames
                else:
                    frames = frames[-self.cfg.frames_per_camera :]
            frames_flat.extend(frames)

        img_stack = np.stack(frames_flat, axis=0)  # (16, H, W, 3)

        # Ego history (world frame)
        pos = np.array(ego_state["position"], dtype=np.float32)
        yaw = float(ego_state["heading"])
        self._ego_pos_hist.append(pos)
        self._ego_yaw_hist.append(yaw)

        T = len(self._ego_pos_hist)
        ego_hist_xyz = np.zeros((1, 1, T, 3), dtype=np.float32)
        ego_hist_yaw = np.zeros((1, 1, T, 1), dtype=np.float32)

        for i, p in enumerate(self._ego_pos_hist):
            ego_hist_xyz[0, 0, i, :] = p
        for i, y in enumerate(self._ego_yaw_hist):
            ego_hist_yaw[0, 0, i, 0] = y

        nav_cmd = None
        for key in ("route_commands", "route_cmds", "nav_cmds", "commands"):
            if isinstance(scenario_data.get(key, None), (list, tuple)) and frame_id < len(scenario_data[key]):
                nav_cmd = scenario_data[key][frame_id]
                break
        if nav_cmd is None:
            nav_cmd = ego_state.get("command", "STRAIGHT")
        nav_cmd = str(nav_cmd)

        if nav_cmd.isdigit():
            n = int(nav_cmd)
            if n in (0, 1):
                nav_cmd = "LEFT"
            elif n in (4, 5):
                nav_cmd = "RIGHT"
            else:
                nav_cmd = "STRAIGHT"

        return {
            "img_stack": img_stack,
            "ego_xyz_world": ego_hist_xyz,
            "ego_yaw_world": ego_hist_yaw,
            "nav_cmd": nav_cmd,
        }

    def run_inference(self, model_input: Any) -> Any:
        if self.client is None:
            raise RuntimeError("AlpamayoR1Adapter: client is None. Did you call load_model()?")

        return self.client.infer(
            image_stack_uint8_nhwc=model_input["img_stack"],
            ego_xyz_world=model_input["ego_xyz_world"],
            ego_rot_yaw=model_input["ego_yaw_world"],
            top_p=self.cfg.top_p,
            temperature=self.cfg.temperature,
            num_traj_samples=self.cfg.num_traj_samples,
            max_generation_length=self.cfg.max_generation_length,
            nav_cmd=model_input.get("nav_cmd", "STRAIGHT"),
            coord_mode=self.cfg.coord_mode,
            cuda_launch_blocking=self.cfg.cuda_launch_blocking,
        )

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        traj_list = model_output.get("trajectory_lateral_forward", None)
        if traj_list is None:
            return {"trajectory": np.zeros((10, 2), dtype=np.float32)}

        traj = np.array(traj_list, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[1] != 2:
            traj = np.zeros((10, 2), dtype=np.float32)

        out: Dict[str, Any] = {"trajectory": traj}

        reasoning = model_output.get("reasoning", None)
        if reasoning is not None:
            out["reasoning"] = str(reasoning)

        return out

    def get_waypoint_dt(self) -> float:
        return float(self.cfg.waypoint_dt)

    def get_trajectory_time_horizon(self) -> float:
        return float(self.cfg.time_horizon_s)