#!/usr/bin/env python3

"""
Alpamayo-specific evaluator tweaks.
Add command + Alpamayo reasoning overlay on the new cam view (cam_f0),
Not changing any of the BaseEvaluator cam_f0 overlay features.
"""

from __future__ import annotations

from pathlib import Path
import textwrap
import cv2

from bridgesim.evaluation.core.base_evaluator import BaseEvaluator


class AlpamayoEvaluator(BaseEvaluator):
    def render_cam_f0_vis(self, env, frame_id, ego_state, plan_traj_ego):

        # Keep camera features from BaseEvaluator (ribbon + route dots + target pole + banner)
        super().render_cam_f0_vis(env, frame_id, ego_state, plan_traj_ego)

        # Add Alpamayo-specific text overlay (cmd + reasoning) without duplicating frame/speed
        frame_output_path = self.output_dir / f"{frame_id:05d}"
        cam_path = frame_output_path / "cam_f0.jpg"
        if not cam_path.exists():
            return

        img = cv2.imread(str(cam_path))
        if img is None:
            return

        cmd_map = {0: "LEFT", 1: "RIGHT", 2: "STRAIGHT", 3: "LANEFOLLOW", -1: "VOID"}
        cmd = int(ego_state.get("command", -1))
        cmd_str = cmd_map.get(cmd, str(cmd))

        reasoning = None
        if isinstance(getattr(self, "cached_parsed_output", None), dict):
            reasoning = self.cached_parsed_output.get("reasoning", None)

        reasoning_clean = None
        if reasoning:
            reasoning_clean = str(reasoning).strip()

        # Do not include frame or speed here bc BaseEvaluator banner already shows it
        lines = [f"cmd={cmd_str}"]

        if reasoning_clean:
            r = " ".join(reasoning_clean.split())
            wrapped = textwrap.wrap(r, width=52)
            lines.append("reasoning:")
            lines.extend(wrapped[:6])

        self._draw_text_panel(
            img,
            lines,
            align="right",
            x=12,
            y=45
        )

        cv2.imwrite(str(cam_path), img)

        try:
            if reasoning:
                (frame_output_path / "reasoning.txt").write_text(str(reasoning))
        except Exception:
            pass

    @staticmethod
    def _draw_text_panel(img, lines, align="right", x=12, y=12):
        """
        Draws a semi-transparent black panel with white text.
        """
        if img is None or not lines:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        line_h = 24
        pad = 10

        H, W = img.shape[:2]

        widths = []
        for ln in lines:
            (w, _), _ = cv2.getTextSize(str(ln), font, scale, thickness)
            widths.append(w)

        box_w = max(widths) + 2 * pad if widths else 200
        box_h = len(lines) * line_h + 2 * pad

        if str(align).lower() == "right":
            x0 = W - x - box_w
        else:
            x0 = x
        y0 = y

        x0 = max(0, min(W - 1, x0))
        y0 = max(0, min(H - 1, y0))
        x1 = max(0, min(W - 1, x0 + box_w))
        y1 = max(0, min(H - 1, y0 + box_h))

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

        yy = y0 + pad + 18
        for ln in lines:
            cv2.putText(img, str(ln), (x0 + pad, yy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            yy += line_h