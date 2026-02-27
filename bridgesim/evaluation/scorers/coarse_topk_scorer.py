import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict

from bridgesim.evaluation.scorers.base_scorer import BaseTrajectoryScorer


class CoarseTopKScorer(BaseTrajectoryScorer):
    """
    Scorer that replicates the full v2 forward_test_rl scoring pipeline
    (lines 1805-1859):

        _get_scorer_inputs → coarse scorer_decoder → score heads →
        topk(32) → fine_scorer_decoder (3 layers) → fine score heads →
        per-layer argmax → map back to global idx → traj_to_score[:, -1]

    For DiffusionDrive v2: uses pre-computed coarse_scores from model output.
    For DiffusionDrive v1: loads all v2 scorer modules from a v2 checkpoint.
    """

    V2_WEIGHT_PREFIX = "_trajectory_head."

    # All scorer module names to extract from v2 checkpoint
    SCORER_MODULE_NAMES = [
        # coarse
        "plan_anchor_scorer_encoder",
        "scorer_decoder",
        "NC_head",
        "EP_head",
        "DAC_head",
        "TTC_head",
        "C_head",
        # fine
        "fine_scorer_decoder",
        "fine_NC_head",
        "fine_EP_head",
        "fine_DAC_head",
        "fine_TTC_head",
        "fine_C_head",
    ]

    def __init__(
        self,
        v2_scorer_checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        topk: int = 32,
    ):
        self.device = device
        self.scorer_modules = None
        self.topk = topk

        if v2_scorer_checkpoint_path is not None:
            self._load_scorer_modules(v2_scorer_checkpoint_path)

    def _load_scorer_modules(self, ckpt_path: str):
        """Load coarse + fine scorer modules from a v2 checkpoint."""
        from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_config import TransfuserConfig
        from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel import (
            ScorerTransformerDecoderLayer,
            ScorerTransformerDecoder,
        )
        from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.modules.blocks import linear_relu_ln

        config = TransfuserConfig()
        d_model = config.tf_d_model  # 256
        d_ffn = config.tf_d_ffn      # 1024
        num_poses = config.trajectory_sampling.num_poses  # 8

        # --- coarse scorer (same as v2 TrajectoryHead lines 1025-1071) ---
        plan_anchor_scorer_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 2 * 512),
            nn.Linear(d_model, 512),
        )
        scorer_decoder_layer = ScorerTransformerDecoderLayer(
            num_poses=num_poses, d_model=d_model, d_ffn=d_ffn, config=config,
        )
        scorer_decoder = ScorerTransformerDecoder(scorer_decoder_layer, 1)

        NC_head = nn.Sequential(*linear_relu_ln(512, 1, 2), nn.Linear(512, 1))
        EP_head = nn.Sequential(*linear_relu_ln(512, 2, 2), nn.Linear(512, 1))
        DAC_head = nn.Sequential(*linear_relu_ln(512, 1, 2), nn.Linear(512, 1))
        TTC_head = nn.Sequential(*linear_relu_ln(512, 1, 2), nn.Linear(512, 1))
        C_head = nn.Sequential(*linear_relu_ln(512, 1, 2), nn.Linear(512, 1))

        # --- fine scorer (same as v2 TrajectoryHead lines 1074-1099) ---
        fine_scorer_decoder_layer = ScorerTransformerDecoderLayer(
            num_poses=num_poses, d_model=d_model, d_ffn=d_ffn, config=config,
        )
        fine_scorer_decoder = ScorerTransformerDecoder(fine_scorer_decoder_layer, 3)

        fine_NC_head = nn.Sequential(*linear_relu_ln(512, 1, 2), nn.Linear(512, 1))
        fine_EP_head = nn.Sequential(*linear_relu_ln(512, 2, 2), nn.Linear(512, 1))
        fine_DAC_head = nn.Sequential(*linear_relu_ln(512, 1, 2), nn.Linear(512, 1))
        fine_TTC_head = nn.Sequential(*linear_relu_ln(512, 1, 2), nn.Linear(512, 1))
        fine_C_head = nn.Sequential(*linear_relu_ln(512, 1, 2), nn.Linear(512, 1))

        self.scorer_modules = nn.ModuleDict({
            "plan_anchor_scorer_encoder": plan_anchor_scorer_encoder,
            "scorer_decoder": scorer_decoder,
            "NC_head": NC_head, "EP_head": EP_head,
            "DAC_head": DAC_head, "TTC_head": TTC_head, "C_head": C_head,
            "fine_scorer_decoder": fine_scorer_decoder,
            "fine_NC_head": fine_NC_head, "fine_EP_head": fine_EP_head,
            "fine_DAC_head": fine_DAC_head, "fine_TTC_head": fine_TTC_head,
            "fine_C_head": fine_C_head,
        })

        # Load weights
        print(f"[CoarseTopKScorer] Loading scorer weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        clean_sd = {}
        for k, v in state_dict.items():
            new_key = k.replace("agent._transfuser_model.", "").replace("_transfuser_model.", "")
            clean_sd[new_key] = v

        scorer_sd = OrderedDict()
        for module_name in self.SCORER_MODULE_NAMES:
            prefix = f"{self.V2_WEIGHT_PREFIX}{module_name}."
            for k, v in clean_sd.items():
                if k.startswith(prefix):
                    local_key = k[len(self.V2_WEIGHT_PREFIX):]
                    scorer_sd[local_key] = v

        missing, unexpected = self.scorer_modules.load_state_dict(scorer_sd, strict=False)
        if missing:
            print(f"[CoarseTopKScorer] Missing keys: {len(missing)}")
            print(f"[CoarseTopKScorer] Sample: {missing[:5]}")
        if unexpected:
            print(f"[CoarseTopKScorer] Unexpected keys: {len(unexpected)}")
            print(f"[CoarseTopKScorer] Sample: {unexpected[:5]}")

        self.scorer_modules.to(self.device)
        self.scorer_modules.eval()
        print("[CoarseTopKScorer] Scorer modules loaded successfully.")

    # ---- v2 normalization (x/50, y/20, heading/1.57) ----

    @staticmethod
    def _v2_norm_odo(traj: torch.Tensor) -> torch.Tensor:
        x = traj[..., 0:1] / 50.0
        y = traj[..., 1:2] / 20.0
        h = traj[..., 2:3] / 1.57
        return torch.cat([x, y, h], dim=-1)

    @staticmethod
    def _v2_denorm_odo(traj: torch.Tensor) -> torch.Tensor:
        x = traj[..., 0:1] * 50.0
        y = traj[..., 1:2] * 20.0
        h = traj[..., 2:3] * 1.57
        return torch.cat([x, y, h], dim=-1)

    # ---- Replicates v2 _get_scorer_inputs (lines 1478-1506) ----

    def _get_scorer_inputs(
        self,
        diffusion_output: torch.Tensor,  # (B, N, 8, 3)
        bs: int,
        ego_fut_mode: int,
    ):
        from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.modules.blocks import (
            gen_sineembed_for_position,
            gen_sineembed_for_position_1d,
        )

        # norm → clamp → denorm
        diffusion_output = self._v2_norm_odo(diffusion_output)
        x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
        noisy_traj_points = self._v2_denorm_odo(x_boxes)  # (B, N, 8, 3)

        noisy_traj_points_xy = noisy_traj_points[..., :2]  # (B, N, 8, 2)

        traj_pos_embed = gen_sineembed_for_position(
            noisy_traj_points_xy, hidden_dim=64
        ).flatten(-2)  # (B, N, 512)

        traj_heading_embed = gen_sineembed_for_position_1d(
            noisy_traj_points[..., 2], hidden_dim=32
        ).flatten(-2)  # (B, N, 512)

        traj_pos_embed = torch.cat([traj_pos_embed, traj_heading_embed], dim=-1)  # (B, N, 1024)

        traj_feature = self.scorer_modules["plan_anchor_scorer_encoder"](traj_pos_embed)
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)  # (B, N, 512)

        return noisy_traj_points_xy, traj_feature, None  # None = time_embed

    # ---- Replicates v2 _select_topk (lines 1508-1554) ----

    @staticmethod
    def _select_topk(
        final_coarse_reward: torch.Tensor,  # (B, N)
        topk: int,
        traj_feature: torch.Tensor,         # (B, N, C)
        noisy_traj_points_xy: torch.Tensor,  # (B, N, 8, 2)
    ):
        actual_k = min(topk, final_coarse_reward.size(1))

        topk_val, topk_idx = torch.topk(
            final_coarse_reward, actual_k, dim=-1, largest=True, sorted=True
        )  # (B, K)

        # gather traj_feature
        idx_feat = topk_idx.unsqueeze(-1).expand(-1, -1, traj_feature.size(-1))  # (B, K, C)
        traj_feature_k = torch.gather(traj_feature, 1, idx_feat)

        # gather noisy_traj_points_xy
        idx_point = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, noisy_traj_points_xy.size(-2), noisy_traj_points_xy.size(-1)
        )  # (B, K, 8, 2)
        noisy_traj_points_k = torch.gather(noisy_traj_points_xy, 1, idx_point)

        return traj_feature_k, noisy_traj_points_k, topk_idx, topk_val

    # ---- Replicates v2 _score_fine_multi (lines 1398-1476, only_reward=True) ----

    def _score_fine_multi(
        self,
        traj_feature_list: List[torch.Tensor],  # list of (B, K, 512) from each decoder layer
    ) -> List[torch.Tensor]:
        """Returns best_idx_list: list of (B,) per fine decoder layer."""
        sigmoid = torch.sigmoid
        best_idx_list = []
        for feat in traj_feature_list:
            NC_score = self.scorer_modules["fine_NC_head"](feat).squeeze(-1)
            EP_score = self.scorer_modules["fine_EP_head"](feat).squeeze(-1)
            DAC_score = self.scorer_modules["fine_DAC_head"](feat).squeeze(-1)
            TTC_score = self.scorer_modules["fine_TTC_head"](feat).squeeze(-1)
            C_score = self.scorer_modules["fine_C_head"](feat).squeeze(-1)
            final_fine_reward = (
                sigmoid(NC_score) * sigmoid(DAC_score)
                * (5 * sigmoid(TTC_score) + 5 * sigmoid(EP_score) + 2 * sigmoid(C_score))
                / 12
            )
            best_idx = torch.argmax(final_fine_reward, dim=-1)  # (B,)
            best_idx_list.append(best_idx)
        return best_idx_list

    # ---- Full pipeline replicating v2 forward_test_rl lines 1805-1859 ----

    def _run_full_pipeline(
        self,
        candidates: torch.Tensor,        # (B, N, 8, 3)  diffusion_output
        scorer_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        bs = candidates.shape[0]
        device = candidates.device

        bev_feature = scorer_context["bev_feature"]
        bev_spatial_shape = scorer_context["bev_spatial_shape"]
        agents_query = scorer_context["agents_query"]
        ego_query = scorer_context["ego_query"]
        status_encoding = scorer_context["status_encoding"]

        # Line 1805: _get_scorer_inputs
        noisy_traj_points_xy, traj_feature, time_embed = self._get_scorer_inputs(
            candidates, bs, candidates.shape[1]
        )

        # Lines 1807-1809: coarse scorer_decoder
        traj_feature_list = self.scorer_modules["scorer_decoder"](
            traj_feature, noisy_traj_points_xy,
            bev_feature, bev_spatial_shape,
            agents_query, ego_query,
            time_embed, status_encoding, None,
        )
        traj_feature = traj_feature_list[-1]  # (B, N, 512)

        # Lines 1810-1815: coarse score heads
        sigmoid = torch.sigmoid
        NC_score = self.scorer_modules["NC_head"](traj_feature).squeeze(-1)
        EP_score = self.scorer_modules["EP_head"](traj_feature).squeeze(-1)
        DAC_score = self.scorer_modules["DAC_head"](traj_feature).squeeze(-1)
        TTC_score = self.scorer_modules["TTC_head"](traj_feature).squeeze(-1)
        C_score = self.scorer_modules["C_head"](traj_feature).squeeze(-1)
        final_coarse_reward = (
            sigmoid(NC_score) * sigmoid(DAC_score)
            * (5 * sigmoid(TTC_score) + 5 * sigmoid(EP_score) + 2 * sigmoid(C_score))
            / 12
        )

        # Lines 1817-1821: best coarse trajectory
        best_coarse_flat = torch.argmax(final_coarse_reward, dim=-1)  # (B,)
        coarse_traj = candidates[
            torch.arange(bs, device=device), best_coarse_flat
        ].unsqueeze(1)  # (B, 1, 8, 3)
        traj_to_score = [coarse_traj]

        # Lines 1823-1829: topk selection
        topk = self.topk
        traj_feature, noisy_traj_points_xy, topk_idx, topk_val = self._select_topk(
            final_coarse_reward, topk, traj_feature, noisy_traj_points_xy,
        )

        # Line 1831: fine scorer_decoder (3 layers)
        fine_traj_feature_list = self.scorer_modules["fine_scorer_decoder"](
            traj_feature, noisy_traj_points_xy,
            bev_feature, bev_spatial_shape,
            agents_query, ego_query,
            time_embed, status_encoding, None,
        )

        # Lines 1832-1834: fine scoring per layer
        best_idx_list = self._score_fine_multi(fine_traj_feature_list)

        # Lines 1835-1840: map local fine best back to global idx
        for best_idx_local in best_idx_list:
            global_best_idx = topk_idx[torch.arange(bs, device=device), best_idx_local]
            fine_traj = candidates[
                torch.arange(bs, device=device), global_best_idx
            ].unsqueeze(1)  # (B, 1, 8, 3)
            traj_to_score.append(fine_traj)

        # Line 1842: concat all
        traj_to_score = torch.cat(traj_to_score, dim=1)  # (B, 4, 8, 3)

        # Lines 1845-1859: build return dict (same keys as v2's cal_pdm=False path)
        topk_trajectories = candidates[
            torch.arange(bs, device=device).unsqueeze(1).expand(-1, topk_idx.size(1)),
            topk_idx,
        ]  # (B, 32, 8, 3)

        return {
            "trajectory": traj_to_score[:, -1],        # (B, 8, 3) fine best from last layer
            "trajectory_candidates": traj_to_score,     # (B, 4, 8, 3)
            "trajectory_topk": topk_trajectories,       # (B, 32, 8, 3)
            "topk_scores": topk_val,                    # (B, 32)
            "trajectory_coarse": candidates,            # (B, N, 8, 3)
            "coarse_scores": final_coarse_reward,       # (B, N)
        }

    def select_best(self, model_output: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Select the best trajectory following v2's full pipeline.

        For v2: uses pre-computed coarse_scores (only coarse argmax, no fine scorer).
        For v1: runs the full coarse → topk → fine pipeline using loaded v2 modules.
        """
        candidates = model_output["all_candidates"]  # (B, N, 8, 3)
        bs = candidates.shape[0]
        device = candidates.device

        coarse_scores = model_output.get("coarse_scores")
        if coarse_scores is not None:
            # V2 path: coarse_scores already computed by the model
            best_idx = coarse_scores.argmax(dim=-1)
            best_traj = candidates[
                torch.arange(bs, device=device), best_idx
            ]
            return {
                "trajectory": best_traj,
                "scores": coarse_scores,
                "best_idx": best_idx,
            }

        # V1 path: run full v2 pipeline (coarse → topk → fine)
        if self.scorer_modules is None:
            raise RuntimeError(
                "No coarse_scores in model output and no scorer modules loaded. "
                "Provide v2_scorer_checkpoint_path when using with DiffusionDrive v1."
            )
        scorer_context = model_output.get("scorer_context")
        if scorer_context is None:
            raise RuntimeError(
                "scorer_context not found in model output. "
                "Ensure forward_inference_scaling returns scorer_context."
            )

        with torch.no_grad():
            result = self._run_full_pipeline(candidates, scorer_context)

        return {
            "trajectory": result["trajectory"].unsqueeze(0) if result["trajectory"].dim() == 2 else result["trajectory"],
            "scores": result["coarse_scores"],
            "best_idx": result["coarse_scores"].argmax(dim=-1),
            "trajectory_candidates": result["trajectory_candidates"],
            "trajectory_topk": result["trajectory_topk"],
            "topk_scores": result["topk_scores"],
            "coarse_scores": result["coarse_scores"],
        }
