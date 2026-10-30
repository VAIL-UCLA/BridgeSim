"""Thin HTTP client for the DreamStream batch inference API.

Dependencies: ``requests``, ``numpy``, ``Pillow`` — all standard.

Usage::

    from bridgesim.evaluation.core.batch_client import BatchClient

    client = BatchClient("http://127.0.0.1:8790")
    info = client.info()

    # Block 0 (optionally with a first-frame anchor image)
    rendered = client.infer(layout_frames[:info["block_0_input_frames"]],
                            prompt="a driving scene", first_frame=anchor)

    # Blocks 1+
    rendered = client.infer(layout_frames[:info["block_input_frames"]])

    # Start a new session
    client.reset()
"""

import base64
import io
from typing import Optional

import numpy as np
import requests
from PIL import Image


class BatchClient:
    """Synchronous HTTP client for the DreamStream batch inference server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8790",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def info(self) -> dict:
        """Query backend metadata and frame count requirements."""
        resp = requests.get(
            f"{self.base_url}/info", timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def infer(
        self,
        layout_frames: list[np.ndarray],
        prompt: Optional[str] = None,
        first_frame: Optional[np.ndarray | str] = None,
    ) -> list[np.ndarray]:
        """Send the layout frames of one block, receive rendered frames.

        Args:
            layout_frames: List of ``(H, W, 3)`` uint8 RGB layout images.
                Block 0 requires ``info()["block_0_input_frames"]`` frames.
                Subsequent blocks require ``info()["block_input_frames"]``.
            prompt: Optional text prompt (overrides server default).
            first_frame: Optional first-frame conditioning image (block 0
                only).  Accepts a ``(H, W, 3)`` uint8 numpy array or a file
                path (PNG/JPEG).

        Returns:
            List of rendered RGB frames as ``(H, W, 3)`` uint8 arrays.
        """
        files = []
        for i, arr in enumerate(layout_frames):
            buf = io.BytesIO()
            np.save(buf, arr)
            buf.seek(0)
            files.append((
                "frames",
                (f"frame_{i}.npy", buf, "application/octet-stream"),
            ))

        data = {}
        if prompt is not None:
            data["prompt"] = prompt

        if first_frame is not None:
            if isinstance(first_frame, str):
                with open(first_frame, "rb") as f:
                    ff_bytes = f.read()
                ext = first_frame.rsplit(".", 1)[-1].lower()
                mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
                files.append(("first_frame", ("first_frame." + ext, ff_bytes, mime)))
            else:
                buf = io.BytesIO()
                Image.fromarray(first_frame).save(buf, format="PNG")
                files.append(("first_frame", ("first_frame.png", buf.getvalue(), "image/png")))

        resp = requests.post(
            f"{self.base_url}/infer",
            files=files,
            data=data,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        result = resp.json()
        decoded: list[np.ndarray] = []
        for b64_str in result["frames"]:
            img = Image.open(io.BytesIO(base64.b64decode(b64_str)))
            decoded.append(np.array(img))

        return decoded

    def reset(self) -> None:
        """Reset server pipeline state (start a new session)."""
        resp = requests.post(
            f"{self.base_url}/reset", timeout=self.timeout,
        )
        resp.raise_for_status()