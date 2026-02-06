from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_config import TransfuserConfig
from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.transfuser_backbone import TransfuserBackbone
from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.transfuser_features import BoundingBox2DIndex
from bridgesim.modelzoo.navsim.common.enums import StateSE2Index
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
import torch.nn.functional as F
from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention, gen_sineembed_for_position_1d, GridSampleCrossBEVAttentionScorer
from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.modules.multimodal_loss import LossComputer
from typing import Any, List, Dict, Optional, Union, Tuple
import math
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
from bridgesim.modelzoo.navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from bridgesim.modelzoo.navsim.common.dataclasses import Trajectory
import lzma
import pickle
import concurrent.futures as cf, cloudpickle, os
import multiprocessing as mp
from bridgesim.modelzoo.navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    WeightedMetricIndex as WIdx,
)
from bridgesim.modelzoo.navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
import joblib

def _pairwise_subscores(scorer):
    """
    从已调用过 score_proposals 的 PDMScorer 中
    拆出 7 个子指标和最终分数，全部 shape=(G,)，顺序与 proposal id 对齐
    返回 dict[str, np.ndarray]
    """
    mm   = scorer._multi_metrics                # (3, N)
    wm   = scorer._weighted_metrics.copy()      # <<< 一定要 copy !
    prod = mm.prod(axis=0)                      # (N,)

    wcoef  = scorer._config.weighted_metrics_array
    thresh = scorer._config.progress_distance_threshold
    prog_raw = scorer._progress_raw             # (N,)

    # ---------- progress 归一化（与 _pairwise_scores 完全一致） ----------
    raw_prog    = prog_raw * prod
    raw_prog_gt = raw_prog[0]
    max_pair    = np.maximum(raw_prog_gt, raw_prog[1:])
    norm_prog   = np.where(
        max_pair > thresh,
        raw_prog[1:] / (max_pair + 1e-6),
        np.where(prod[1:] == 0.0, 0.0, 1.0),
    ).astype(np.float64)
    wm[WeightedMetricIndex.PROGRESS, 1:] = norm_prog

    # ---------- 加权指标 ----------
    wscore = (wm * wcoef[:, None]).sum(axis=0) / wcoef.sum()

    return {
        "no_collision"  : mm[MultiMetricIndex.NO_COLLISION,        1:].copy(),
        "drivable_area" : mm[MultiMetricIndex.DRIVABLE_AREA,       1:].copy(), # 乘法版
        "progress"      : wm[WeightedMetricIndex.PROGRESS,         1:].copy(),
        "ttc"           : wm[WeightedMetricIndex.TTC,              1:].copy(),
        "comfort"       : wm[WeightedMetricIndex.COMFORTABLE,      1:].copy(),
        "dir_weighted"  : wm[WeightedMetricIndex.DRIVING_DIRECTION,1:].copy(),
        "final"         : prod[1:] * wscore[1:],                               # 总分
    }
def _pairwise_scores(scorer) -> np.ndarray:
    """
    使用 scorer 在 batch 模式下缓存的中间结果，
    重新计算“GT (索引0) vs 每条候选”的得分。
    返回 shape = (N-1,)  float32。
    """
    # --- 取中间量 ---------------------------------------------------
    mm   = scorer._multi_metrics            # (M_mul, N)
    wm   = scorer._weighted_metrics.copy()  # (M_wgt, N)  (复制以便我们改进程)
    prog_raw = scorer._progress_raw         # (N,)
    weight_coef = scorer._config.weighted_metrics_array  # (M_wgt,)

    N = mm.shape[1]                         # proposals = 1(GT) + G
    assert N >= 2, "Need at least GT + 1 proposal"

    # --- 计算乘法指标乘积 ------------------------------------------
    multi_prod = mm.prod(axis=0)            # (N,)

    # --- 重新归一化 progress，每条候选只与 GT 对标 ------------------
    raw_prog    = prog_raw * multi_prod     # (N,)
    raw_prog_gt = raw_prog[0]

    max_pair    = np.maximum(raw_prog_gt, raw_prog[1:])           # (G,)
    thresh      = scorer._config.progress_distance_threshold

    # 若 max_pair > thresh → 按比例归一；否则看 collision 情况
    norm_prog   = np.where(
        max_pair > thresh,
        raw_prog[1:] / (max_pair + 1e-6),
        np.where(multi_prod[1:] == 0.0, 0.0, 1.0),
    ).astype(np.float64)                                         # (G,)

    # 把 progress 行（WeightedMetricIndex.PROGRESS）替换成新的
    wm[WIdx.PROGRESS, 1:] = norm_prog

    # --- 计算 weighted_metric_scores（与 _aggregate_scores 同式） ----
    weighted_scores = (wm[:, 1:] * weight_coef[:, None]).sum(axis=0)
    weighted_scores /= weight_coef.sum()                         # (G,)

    # --- 最终得分 = 乘法指标 × 加权指标 -----------------------------
    final_scores = multi_prod[1:] * weighted_scores              # (G,)

    return final_scores.astype(np.float32)                       # (G,)


def _pdm_worker(args):
    cache, traj_np = args
    # if isinstance(cache, str): 
    with lzma.open(cache, "rb") as f:
        metric_cache = pickle.load(f)
    # else:
    #     metric_cache = cache
    results, sim_traj = pdm_score_para(
        metric_cache=metric_cache,
        model_trajectory=traj_np,                # (G, T, C)
        future_sampling=SIMULATOR.proposal_sampling,
        simulator=SIMULATOR,                    # 全局对象，见 initializer
        scorer=SCORER,
    )
    scores = _pairwise_scores(SCORER)
    subscores  = _pairwise_subscores(SCORER)
    return scores.astype(np.float32), metric_cache, subscores, sim_traj.astype(np.float32)  # (G,)

def _init_pool(sim_cfg, scorer_cfg):
    global SIMULATOR, SCORER
    SIMULATOR = instantiate(sim_cfg)
    SCORER    = instantiate(scorer_cfg)

class V2TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )

        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1,320),
        )


    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None, eta=0.0, metric_cache=None, cal_pdm=True,token=None, use_bev_calibrator=False) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        if use_bev_calibrator and hasattr(self, 'bev_calibrator') and self.bev_calibrator is not None:
            # With calibration: get bev_feature, calibrate, then derive bev_feature_upscale via top_down
            _, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
            bev_feature = self.bev_calibrator.calibrate(bev_feature)
            bev_feature_upscale = self._backbone.top_down(bev_feature)
        else:
            # Without calibration: use backbone output directly
            bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)

        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        # upsample to the same shape as bev_feature_upscale

        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        # concat concat_cross_bev and cross_bev_feature
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))
        cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        pred = self._trajectory_head(trajectory_query,agents_query, cross_bev_feature,bev_spatial_shape,status_encoding[:, None],status_feature,camera_feature,targets=targets,global_img=None,eta=eta,metric_cache=metric_cache, cal_pdm=cal_pdm,token=token)
        output.update(pred)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output

    def forward_temporal(
        self,
        features: Dict[str, torch.Tensor],
        current_time: float,
        targets: Dict[str, torch.Tensor] = None,
        eta=0.0,
        metric_cache=None,
        cal_pdm=True,
        token=None,
        use_bev_calibrator=False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with temporal consistency scoring.

        This method is similar to the standard forward but uses the temporal
        consistency scorer for coarse trajectory selection.

        Args:
            features: Input features dict with camera_feature, lidar_feature, status_feature
            current_time: Current simulation time in seconds (for temporal consistency)
            targets: Training targets (unused in inference)
            eta: Diffusion eta parameter
            metric_cache: PDM metric cache
            cal_pdm: Whether to calculate PDM scores
            token: Token identifier
            use_bev_calibrator: Whether to apply BEV calibration for domain adaptation

        Returns:
            Output dictionary with trajectory and scoring information
        """
        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        if use_bev_calibrator and hasattr(self, 'bev_calibrator') and self.bev_calibrator is not None:
            # With calibration: get bev_feature, calibrate, then derive bev_feature_upscale via top_down
            _, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
            bev_feature = self.bev_calibrator.calibrate(bev_feature)
            bev_feature_upscale = self._backbone.top_down(bev_feature)
        else:
            # Without calibration: use backbone output directly
            bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)

        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:, :-1].permute(0, 2, 1).contiguous().view(
            batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1]
        )
        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2, -1).permute(0, 2, 1))
        cross_bev_feature = cross_bev_feature.permute(0, 2, 1).contiguous().view(
            batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1]
        )
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        # Use temporal consistency forward pass
        pred = self._trajectory_head.forward_test_rl_temporal_scorer(
            trajectory_query,
            agents_query,
            cross_bev_feature,
            bev_spatial_shape,
            status_encoding[:, None],
            status_feature,
            camera_feature,
            targets=targets,
            global_img=None,
            metric_cache=metric_cache,
            current_time=current_time,
            eta=eta,
            cal_pdm=cal_pdm,
            token=token,
        )
        output.update(pred)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output

    def reset_temporal_history(self):
        """Reset temporal consistency history. Call at start of new scenario."""
        self._trajectory_head.reset_temporal_history()

    def set_temporal_params(self, alpha: float = None, lambda_consist: float = None,
                           max_history: int = None, sigma: float = None,
                           consensus_temperature: float = None):
        """Set temporal consistency parameters."""
        self._trajectory_head.set_temporal_params(alpha, lambda_consist, max_history, sigma, consensus_temperature)

    def get_temporal_history_length(self) -> int:
        """Get current length of temporal history buffer."""
        return len(self._trajectory_head.temporal_history)

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
    def forward(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape
        # 6. get final prediction
        traj_feature = traj_feature.view(bs, ego_fut_mode,-1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs,ego_fut_mode, self.ego_fut_ts, 3)

        return plan_reg, plan_cls
class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                    global_cond, time_embed
                ], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
            global_feature = torch.cat([
                    global_img, global_feature
                ], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale,shift = scale_shift.chunk(2,dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature

class ScorerTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        tf_d_model: int = 512
        tf_d_ffn: int = 2048
        tf_num_layers: int = 6
        tf_num_head: int = 16
        tf_dropout: float = 0.1

        self.cross_bev_attention = GridSampleCrossBEVAttentionScorer(
            tf_d_model,
            tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )
        self.agent_input = nn.Linear(256, tf_d_model)
        self.ego_input = nn.Linear(256, tf_d_model)
        self.cross_agent_attention = nn.MultiheadAttention(
            tf_d_model,
            tf_num_head,
            dropout=tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            tf_d_model,
            tf_num_head,
            dropout=tf_dropout,
            batch_first=True,
        )
        self.self_attn = nn.MultiheadAttention(
            tf_d_model, tf_num_head,
            dropout=tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(tf_d_model, tf_d_ffn),
            nn.ReLU(),
            nn.Linear(tf_d_ffn, tf_d_model),
        )
        self.norm1 = nn.LayerNorm(tf_d_model)
        self.norm2 = nn.LayerNorm(tf_d_model)
        self.norm3 = nn.LayerNorm(tf_d_model)
        self.norm4 = nn.LayerNorm(tf_d_model)
        # self.time_modulation = ModulationLayer(config.tf_d_model,256)

    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
        agents_query = self.agent_input(agents_query)
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query,agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        
        traj_feature = traj_feature + self.dropout1(self.self_attn(traj_feature, traj_feature, traj_feature)[0])
        traj_feature = self.norm2(traj_feature)

        # 4.5 cross attention with  ego query
        ego_query = self.ego_input(ego_query)
        traj_feature = traj_feature + self.dropout2(self.cross_ego_attention(traj_feature, ego_query,ego_query)[0])
        traj_feature = self.norm3(traj_feature)

        # 4.6 feedforward network
        traj_feature = self.norm4(self.ffn(traj_feature))

        return traj_feature
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class ScorerTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        traj_feature_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            traj_feature = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            traj_feature_list.append(traj_feature)
        return traj_feature_list


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.time_modulation = ModulationLayer(config.tf_d_model,256)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=20,
        )

    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query,agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        
        # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])

        # 4.5 cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query,ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        
        # 4.6 feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        # 4.8 modulate with time steps
        traj_feature = self.time_modulation(traj_feature, time_embed,global_cond=None,global_img=global_img)
        
        # 4.9 predict the offset & heading
        traj_feature = traj_feature.view(traj_feature.shape[0], -1, 20, traj_feature.shape[-1])
        bs,num_groups, _, _ = traj_feature.shape
        traj_feature = traj_feature.view(-1, 20, traj_feature.shape[-1])
        poses_reg, poses_cls = self.task_decoder(traj_feature) #bs,20,8,3; bs,20
        poses_reg = poses_reg.view(bs, 20*num_groups, 8, 3)
        poses_cls = poses_cls.view(bs, -1, 20)
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return poses_reg, poses_cls, traj_feature
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls, traj_feature = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[...,:2].clone().detach()
        return poses_reg_list, poses_cls_list, traj_feature

class DDIMScheduler_with_logprob(DDIMScheduler):
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 1.0, # 1.0 for ddpm, 0.0 for ddim
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        prev_sample: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = (
            timestep - self.config.num_train_timesteps // self.num_inference_steps
        )
        # # to prevent OOB on gather
        # prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)
        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = (eta * variance ** (0.5)).clamp_(min=1e-10)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2).clamp_(min=0) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if prev_sample_mean is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
                " `prev_sample` stays `None`."
            )
        if eta > 0:
            std_dev_t_mul = torch.clip(std_dev_t, min=0.04)
            std_dev_t_add = torch.tensor(0.0).to(std_dev_t.device)
        else:
            std_dev_t_mul = torch.tensor(0.0).to(std_dev_t.device)
            std_dev_t_add = torch.tensor(0.0).to(std_dev_t.device)
        if prev_sample is None:
            # 乘性噪声
            variance_noise_horizon = randn_tensor(
                [model_output.shape[0],model_output.shape[1],1,1], generator=generator, device=model_output.device, dtype=model_output.dtype
            ) * std_dev_t_mul + 1.0
            variance_noise_vert = randn_tensor(
                [model_output.shape[0],model_output.shape[1],1,1], generator=generator, device=model_output.device, dtype=model_output.dtype
            ) * std_dev_t_mul + 1.0

            variance_noise_mul = torch.cat((variance_noise_horizon,variance_noise_vert),dim=-1)
            variance_noise_mul = variance_noise_mul.repeat(1,1,model_output.shape[2],1)

            # 加性噪声
            variance_noise_x = randn_tensor(
                [model_output.shape[0],model_output.shape[1],1,1], generator=generator, device=model_output.device, dtype=model_output.dtype
            )
            variance_noise_y = randn_tensor(
                [model_output.shape[0],model_output.shape[1],1,1], generator=generator, device=model_output.device, dtype=model_output.dtype
            )
            variance_noise_add = torch.cat((variance_noise_x,variance_noise_y),dim=-1)
            variance_noise_add = variance_noise_add.repeat(1,1,model_output.shape[2],1)

            prev_sample = prev_sample_mean * variance_noise_mul + std_dev_t_add * variance_noise_add
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t_mul**2))
            - torch.log(std_dev_t_mul)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )   
        log_prob = log_prob.sum(dim=(-2, -1))
        return prev_sample.type(sample.dtype), log_prob, prev_sample_mean.type(sample.dtype)

class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str,config: TransfuserConfig):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            steps_offset=1,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )
        self.diffusionrl_scheduler = DDIMScheduler_with_logprob(
            num_train_timesteps=1000,
            steps_offset=1,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )
        self.num_groups = config.num_groups
        plan_anchor = np.load(plan_anchor_path)

        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) # 20,8,2
        self.sigmoid = nn.Sigmoid()
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1,512),
            nn.Linear(d_model, d_model),
        )
        self.plan_anchor_scorer_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1,2*512),
            nn.Linear(d_model, 512),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 1)
        
        # coarse scorer
        scorer_decoder_layer = ScorerTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.scorer_decoder = ScorerTransformerDecoder(scorer_decoder_layer, 1)
        self.NC_head = nn.Sequential(
            *linear_relu_ln(512, 1, 2),
            nn.Linear(512, 1),
        )
        self.EP_head = nn.Sequential(
            *linear_relu_ln(512, 2, 2),
            nn.Linear(512, 1),
        )
        self.DAC_head = nn.Sequential(
            *linear_relu_ln(512, 1, 2),
            nn.Linear(512, 1),
        )
        self.TTC_head = nn.Sequential(
            *linear_relu_ln(512, 1, 2),
            nn.Linear(512, 1),
        )
        self.C_head = nn.Sequential(
            *linear_relu_ln(512, 1, 2),
            nn.Linear(512, 1),
        )

        # fine scorer
        fine_scorer_decoder_layer = ScorerTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.fine_scorer_decoder = ScorerTransformerDecoder(fine_scorer_decoder_layer, 3)
        self.fine_NC_head = nn.Sequential(
            *linear_relu_ln(512, 1, 2),
            nn.Linear(512, 1),
        )
        self.fine_EP_head = nn.Sequential(
            *linear_relu_ln(512, 2, 2),
            nn.Linear(512, 1),
        )
        self.fine_DAC_head = nn.Sequential(
            *linear_relu_ln(512, 1, 2),
            nn.Linear(512, 1),
        )
        self.fine_TTC_head = nn.Sequential(
            *linear_relu_ln(512, 1, 2),
            nn.Linear(512, 1),
        )
        self.fine_C_head = nn.Sequential(
            *linear_relu_ln(512, 1, 2),
            nn.Linear(512, 1),
        )

        self.rank_loss = torch.nn.MarginRankingLoss(margin=0.05)
        self.loss_computer = LossComputer(config)
        self.targets = [] 
        self.num_draw = 0

        # pdm score - only initialize if config file exists (not needed for inference)
        self._pdm_pool = None
        self.simulator = None
        self.scorer = None
        self.simulator_cfg = None
        self.scorer_cfg = None
        self.metric_caches = {}

        self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_bce_without_reduce = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_reg = nn.MSELoss()
        self.diffusion_output = None

        self.vocab_pdm_gt_path = 'gtrs_traj/navtrain_16384.pkl'
        self.vocab_path = 'gtrs_traj/16384.npy'
        self.vocab_pdm_score_full = None
        self.vocab = None

        # Temporal consistency scorer parameters
        self.temporal_history = []  # List of (sim_time, consensus_trajectory) tuples
        self.temporal_alpha = 1.5  # Decay base: older trajectories get more weight
        self.temporal_lambda = 0.3  # Weight for temporal consistency in combined score
        self.temporal_max_history = 8  # Max number of past trajectories to store
        self.temporal_sigma = 5.0  # Position normalization factor (meters)
        self.temporal_traj_dt = 0.5  # Time between trajectory waypoints (seconds)
        self.consensus_temperature = 1.0  # Softmax temperature for consensus weighting

    def reset_temporal_history(self):
        """Reset the temporal consistency history buffer. Call at start of new scenario."""
        self.temporal_history = []

    def set_temporal_params(self, alpha: float = None, lambda_consist: float = None,
                           max_history: int = None, sigma: float = None,
                           consensus_temperature: float = None):
        """Set temporal consistency parameters."""
        if alpha is not None:
            self.temporal_alpha = alpha
        if lambda_consist is not None:
            self.temporal_lambda = lambda_consist
        if max_history is not None:
            self.temporal_max_history = max_history
        if sigma is not None:
            self.temporal_sigma = sigma
        if consensus_temperature is not None:
            self.consensus_temperature = consensus_temperature

    def compute_consensus_trajectory(
        self,
        trajectories: torch.Tensor,  # (B, G, 8, 2/3) all candidate trajectories
        pdm_scores: torch.Tensor,    # (B, G) PDM scores (NOT combined scores)
    ) -> torch.Tensor:
        """
        Compute score-weighted consensus trajectory.

        The consensus trajectory represents the model's "confident belief" about
        the future, weighted by PDM scores. Higher-scored trajectories contribute
        more to the consensus.

        Mathematical formulation:
            weights = softmax(pdm_scores / temperature)
            consensus = Σ (weights_k * trajectory_k)

        Args:
            trajectories: (B, G, 8, 2/3) all candidate trajectories in ego frame
            pdm_scores: (B, G) PDM scores for weighting (use PDM only, not combined
                        scores, to avoid circular dependency with temporal consistency)

        Returns:
            consensus: (B, 8, 2) score-weighted average trajectory
        """
        # Compute softmax weights from PDM scores
        weights = torch.nn.functional.softmax(
            pdm_scores / self.consensus_temperature, dim=1
        )  # (B, G)

        # Expand weights for broadcasting: (B, G) -> (B, G, 1, 1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        # Compute weighted average trajectory (only x, y coordinates)
        consensus = (trajectories[..., :2] * weights).sum(dim=1)  # (B, 8, 2)

        return consensus

    def compute_temporal_consistency_score(
        self,
        candidates: torch.Tensor,  # (B, G, 8, 2) or (B, G, 8, 3) - candidates in ego frame
        current_time: float,
    ) -> torch.Tensor:
        """
        Compute temporal consistency scores for candidate trajectories.

        The score measures how well each candidate aligns with where previous
        pseudo-expert trajectories predicted the ego would be at the current time.
        Older predictions are weighted more heavily (they were more in-distribution).

        Uses exponential decay: score = exp(-displacement / sigma) to produce
        scores in (0, 1] range where:
          - displacement = 0  -> score = 1.0 (perfect consistency)
          - displacement >> sigma -> score -> 0 (poor consistency)

        Args:
            candidates: (B, G, 8, 2/3) candidate trajectories in ego frame
            current_time: Current simulation time in seconds

        Returns:
            consistency_scores: (B, G) scores in (0, 1] where higher = more consistent
        """
        B, G = candidates.shape[:2]
        device = candidates.device

        # If no history, return ones (no penalty, full score)
        if len(self.temporal_history) == 0:
            return torch.ones(B, G, device=device)

        # Get current position of candidates (first waypoint at t + traj_dt)
        # candidates[..., :2] gives (x, y) positions
        cand_first_pos = candidates[:, :, 0, :2]  # (B, G, 2)

        score_sum = torch.zeros(B, G, device=device)
        weight_sum = 0.0

        for j, (hist_time, hist_traj) in enumerate(self.temporal_history):
            # How many steps ago was this trajectory planned?
            steps_ago = len(self.temporal_history) - j  # 1 = most recent, larger = older
            # history_len = len(self.temporal_history)
            # With alpha < 1: older trajectories get larger weight, all weights in (0, alpha]
            # e.g., alpha=0.9, history_len=10: oldest -> 0.9^1=0.9, newest -> 0.9^10=0.35
            # weight = self.temporal_alpha ** (history_len + 1 - steps_ago)
            weight = self.temporal_alpha ** (steps_ago)

            # Where did hist_traj predict we'd be at current_time?
            # hist_traj is in ego frame at hist_time, shape (8, 2) or (8, 3)
            time_diff = current_time - hist_time
            waypoint_idx = int(time_diff / self.temporal_traj_dt) - 1  # -1 because waypoint 0 is at t+dt

            if 0 <= waypoint_idx < hist_traj.shape[0]:
                # Get predicted position from history
                predicted_pos = hist_traj[waypoint_idx, :2]  # (2,)
                if isinstance(predicted_pos, np.ndarray):
                    predicted_pos = torch.from_numpy(predicted_pos).to(device)

                # Compute displacement from each candidate's first waypoint to predicted position
                # Note: This assumes we're comparing in the same coordinate frame
                # In practice, we need to transform hist_traj to current ego frame
                # For simplicity, we compare the absolute positions assuming small drift
                displacement = torch.norm(cand_first_pos - predicted_pos.view(1, 1, 2), dim=-1)  # (B, G)

                # Convert displacement to score using exponential decay: (0, 1]
                # score = exp(-displacement / sigma)
                position_score = torch.exp(-displacement / self.temporal_sigma)
                score_sum += weight * position_score
                weight_sum += weight

        if weight_sum > 0:
            # Weighted average of position scores, already in (0, 1]
            consistency_scores = score_sum / weight_sum
        else:
            consistency_scores = torch.ones(B, G, device=device)

        return consistency_scores

    def update_temporal_history(
        self,
        selected_traj: np.ndarray,  # (8, 2) or (8, 3) selected trajectory in ego frame
        current_time: float,
    ):
        """
        Add selected trajectory to history buffer.

        Args:
            selected_traj: The trajectory that was selected at this timestep
            current_time: Current simulation time in seconds
        """
        # Store a copy to avoid mutation issues
        traj_copy = selected_traj.copy() if isinstance(selected_traj, np.ndarray) else selected_traj.cpu().numpy()
        self.temporal_history.append((current_time, traj_copy))

        # Trim to max history
        if len(self.temporal_history) > self.temporal_max_history:
            self.temporal_history.pop(0)

    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = odo_info_fut_x/50
        odo_info_fut_y = odo_info_fut_y/20
        odo_info_fut_head = odo_info_fut_head/1.57 # not used
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = odo_info_fut_x * 50
        odo_info_fut_y = odo_info_fut_y * 20
        odo_info_fut_head = odo_info_fut_head*1.57 # not used
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)


    def forward(self, ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,status_feature,camera_feature, targets=None,global_img=None,eta=0.0,metric_cache=None, cal_pdm=True, old_model=False,token=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.EP_head.training:
            return self.forward_train_rl(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,status_feature,camera_feature,targets,global_img,eta,metric_cache,cal_pdm=cal_pdm,token=token)
        else:
            return self.forward_test_rl(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,status_feature,camera_feature,targets,global_img,metric_cache,eta,cal_pdm=cal_pdm,token=token)

    def get_pdm_score_para(self, trajectory, metric_cache_path):
        B, G = trajectory.shape[:2]
        traj_np = trajectory.detach().cpu().numpy()
        futures = [
            self._pdm_pool.submit(
                _pdm_worker,
                (metric_cache_path[b], traj_np[b]),
            )
            for b in range(B)
        ]
        scores_np = np.vstack([f.result()[0] for f in futures])    # (B,G)
        metric_cache = [f.result()[1] for f in futures]
        sub_scores  = [f.result()[2] for f in futures]
        sim_traj = [f.result()[3] for f in futures]
        sim_traj = np.stack(sim_traj, axis=0)
        return torch.from_numpy(scores_np).to(trajectory.device), metric_cache, sub_scores, sim_traj

    def _score_coarse(
        self,
        traj_feature: torch.Tensor,                 # (B, Gk, C)   k==num_groups*ego_fut_mode
        sub_rewards_group: Dict[str, torch.Tensor], # 每个 shape=(B, Gk)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回: loss_coarse, final_coarse_reward, coarse_reward"""
        bs = traj_feature.shape[0]  # batch size
        NC_score  = self.NC_head(traj_feature).squeeze(-1)   # (B,Gk)
        EP_score  = self.EP_head(traj_feature).squeeze(-1)
        DAC_score = self.DAC_head(traj_feature).squeeze(-1)
        TTC_score = self.TTC_head(traj_feature).squeeze(-1)
        C_score   = self.C_head(traj_feature).squeeze(-1)
        gt_nc = sub_rewards_group["no_collision"]
        gt_nc[gt_nc == 0.5] = 0.0

        loss_nc = self.loss_bce(NC_score, gt_nc)
        loss_ep = self.loss_bce(EP_score, sub_rewards_group["progress"])
        loss_dac = self.loss_bce(DAC_score, sub_rewards_group["drivable_area"])
        loss_ttc = self.loss_bce(TTC_score, sub_rewards_group["ttc"])
        loss_c   = self.loss_bce_without_reduce(C_score, sub_rewards_group["comfort"])
        mask = (sub_rewards_group["comfort"] != -1)
        loss_c = (loss_c * mask).sum() / (mask.sum() + 1e-6)

        # ---------- ② EP → Margin‑Rank loss ----------
        gt_ep = sub_rewards_group["progress"]           # (B, Gk)

        B, Gk = EP_score.shape
        idx_i, idx_j = torch.combinations(               # 生成两两组合索引
            torch.arange(Gk, device=EP_score.device), r=2
        ).unbind(-1)                                     # 形状 (P,)

        pred_i, pred_j = EP_score[:, idx_i], EP_score[:, idx_j]   # (B, P)
        gt_i  , gt_j   = gt_ep[:, idx_i], gt_ep[:, idx_j]   # (B, P)

        target = torch.sign(gt_i - gt_j)                          # 1 / -1 / 0
        mask   = target != 0                                      # 排除相等
        if mask.any():
            loss_rank = self.rank_loss(
                pred_i[mask], pred_j[mask], target[mask]
            )
        else:                                                     # 全部相等时不计入
            loss_rank = torch.tensor(0., device=EP_score.device)

        loss_coarse = loss_nc + loss_ep + loss_dac + loss_ttc + loss_c + 2*loss_rank

        loss_dict = {
            "coarse_loss_nc": loss_nc,
            "coarse_loss_ep": loss_ep,
            "coarse_loss_dac": loss_dac,
            "coarse_loss_ttc": loss_ttc,
            "coarse_loss_c": loss_c,
            "coarse_loss_rank": loss_rank,
        }

        final_coarse_reward = (
            self.sigmoid(NC_score) * self.sigmoid(DAC_score) *
            (5 * self.sigmoid(TTC_score) +
            5 * self.sigmoid(EP_score)  +
            2 * self.sigmoid(C_score)) / 12
        )                                                   # (B,Gk)

        best_idx        = torch.argmax(final_coarse_reward, dim=-1)   # (B,)
        # reward_flat     = sub_rewards_group['final'].reshape(bs, num_groups * ego_fut_mode)
        coarse_reward   = sub_rewards_group['final'][torch.arange(bs), best_idx]     # (B,)
        return loss_coarse, final_coarse_reward, coarse_reward, loss_dict

    def _score_fine_multi(
        self,
        traj_feature_list: torch.Tensor,                 # (B, Gk, C)   k==num_groups*ego_fut_mode
        sub_rewards_group: Dict[str, torch.Tensor], # 每个 shape=(B, Gk)
        only_reward: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_fine = 0.0
        loss_dict = {}
        fine_reward_dict = {}
        fine_reward = 0.0
        best_idx_list = []
        bs = traj_feature_list[0].shape[0]
        if sub_rewards_group is not None:
            gt_nc = sub_rewards_group["no_collision"]
            gt_nc[gt_nc == 0.5] = 0.0
        for i, feat in enumerate(traj_feature_list):
            EP_score = self.fine_EP_head(feat).squeeze(-1)
            NC_score = self.fine_NC_head(feat).squeeze(-1)
            DAC_score = self.fine_DAC_head(feat).squeeze(-1)
            TTC_score = self.fine_TTC_head(feat).squeeze(-1)
            C_score = self.fine_C_head(feat).squeeze(-1)
            if not only_reward:
                loss_nc = self.loss_bce(NC_score, gt_nc)
                loss_ep = self.loss_bce(EP_score, sub_rewards_group["progress"])
                # loss_ep = F.smooth_l1_loss(self.sigmoid(EP_score), sub_rewards_group["progress"])
                loss_dac = self.loss_bce(DAC_score, sub_rewards_group["drivable_area"])
                loss_ttc = self.loss_bce(TTC_score, sub_rewards_group["ttc"])
                loss_c   = self.loss_bce_without_reduce(C_score, sub_rewards_group["comfort"])
                mask = (sub_rewards_group["comfort"] != -1)
                loss_c = (loss_c * mask).sum() / (mask.sum() + 1e-6)

                # ---------- ② EP → Margin‑Rank loss ----------
                gt_ep = sub_rewards_group["progress"]           # (B, Gk)

                B, Gk = EP_score.shape
                idx_i, idx_j = torch.combinations(               # 生成两两组合索引
                    torch.arange(Gk, device=EP_score.device), r=2
                ).unbind(-1)                                     # 形状 (P,)

                pred_i, pred_j = EP_score[:, idx_i], EP_score[:, idx_j]   # (B, P)
                gt_i  , gt_j   = gt_ep[:, idx_i], gt_ep[:, idx_j]   # (B, P)

                target = torch.sign(gt_i - gt_j)                          # 1 / -1 / 0
                mask   = target != 0                                      # 排除相等

                if mask.any():
                    loss_rank = self.rank_loss(
                        pred_i[mask], pred_j[mask], target[mask]
                    )
                else:                                                     # 全部相等时不计入
                    loss_rank = torch.tensor(0., device=EP_score.device)
                loss_fine_ = loss_nc + loss_ep + loss_dac + loss_ttc + loss_c + 2*loss_rank
                loss_dict.update({
                    f"fine_loss_nc_{i}": loss_nc,
                    f"fine_loss_ep_{i}": loss_ep,
                    f"fine_loss_dac_{i}": loss_dac,
                    f"fine_loss_ttc_{i}": loss_ttc,
                    f"fine_loss_c_{i}": loss_c,
                    f"fine_loss_rank_{i}": loss_rank,
                })

                loss_fine = loss_fine + loss_fine_
            final_fine_reward = (
                self.sigmoid(NC_score) * self.sigmoid(DAC_score) *
                (5 * self.sigmoid(TTC_score) +
                5 * self.sigmoid(EP_score)  +
                2 * self.sigmoid(C_score)) / 12
            )  
            best_idx        = torch.argmax(final_fine_reward, dim=-1)   # (B,)
            best_idx_list.append(best_idx)
            if not only_reward:
                fine_reward   = sub_rewards_group['final'][torch.arange(bs), best_idx]     # (B,)
                fine_reward_dict.update({
                    f"fine_reward_{i}": fine_reward.mean(),
                })

        loss_fine = loss_fine / len(traj_feature_list)       # 取平均        

        return loss_fine, final_fine_reward, fine_reward, loss_dict, fine_reward_dict, best_idx_list

    def _get_scorer_inputs(self,
                            diffusion_output: torch.Tensor,   # (B, G_all, 8, 3)  —— 已 bezier/denorm
                            bs: int,
                            ego_fut_mode: int):
        """
        返回：
            noisy_traj_points_xy, traj_feature, time_embed
        变量命名和类型都保持不变，直接在 forward 中调用：
            noisy_xy, traj_feature, time_embed = self._make_scorer_inputs(diffusion_output, bs, ego_fut_mode)
        """
        # ---------- ① 轨迹归一化 / 反归一化 ----------
        diffusion_output = self.norm_odo(diffusion_output)
        x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
        noisy_traj_points = self.denorm_odo(x_boxes)                  # (B,G,8,3)

        # ---------- ② xy + heading 的位置编码 ----------
        noisy_traj_points_xy = noisy_traj_points[..., :2]
        traj_pos_embed = gen_sineembed_for_position(
                            noisy_traj_points_xy, hidden_dim=64
                        ).flatten(-2)                                # (B,G,8*64)
        traj_heading_embed = gen_sineembed_for_position_1d(
                                noisy_traj_points[..., 2], hidden_dim=32
                            ).flatten(-2)                            # (B,G,8*32)

        traj_pos_embed = torch.cat([traj_pos_embed, traj_heading_embed], dim=-1)
        traj_feature   = self.plan_anchor_scorer_encoder(traj_pos_embed)   # (B,G,C_raw)
        traj_feature   = traj_feature.view(bs, ego_fut_mode, -1)           # (B,G,C)

        return noisy_traj_points_xy, traj_feature, None

    def _select_topk(self,
                    final_coarse_reward: torch.Tensor,   # (B, G_all)
                    topk: int,
                    traj_feature: torch.Tensor,          # (B, G_all, C)
                    noisy_traj_points_xy: torch.Tensor,  # (B, G_all, 8, 2)
                    sub_rewards_group: Dict[str, torch.Tensor],  # 每个 (B, G_all)
                    ) -> Tuple[torch.Tensor, torch.Tensor,
                                Dict[str, torch.Tensor],
                                torch.Tensor, torch.Tensor]:
        """
        返回：
            traj_feature_k        (B, k, C)
            noisy_traj_points_k   (B, k, 8, 2)
            sub_rewards_topk      dict,每个 (B, k)
            topk_idx              (B, k)   —— 原始全局 idx
            topk_val              (B, k)   —— 对应 coarse 分数
        """
        bs = final_coarse_reward.size(0)

        # ---------- ① 取 Top-k 索引 ----------
        topk_val, topk_idx = torch.topk(
            final_coarse_reward, topk, dim=-1, largest=True, sorted=True
        )                                               # (B,k)

        # ---------- ② 同步裁剪特征 ----------
        idx_feat  = topk_idx.unsqueeze(-1) \
                            .expand(-1, -1, traj_feature.size(-1))        # (B,k,C)
        traj_feature_k = torch.gather(traj_feature, 1, idx_feat)          # (B,k,C)

        idx_point = topk_idx.unsqueeze(-1).unsqueeze(-1) \
                                .expand(-1, -1,
                                        noisy_traj_points_xy.size(-2),
                                        noisy_traj_points_xy.size(-1))     # (B,k,8,2)
        noisy_traj_points_k = torch.gather(noisy_traj_points_xy, 1, idx_point)

        # ---------- ③ 扁平化 / 整理 ----------
        # 这里保持 (B,k,…) 形状，后面如需再 reshape 自行处理
        if sub_rewards_group is not None:
            sub_rewards_topk = {
                name: torch.gather(val, 1, topk_idx)        # (B,k)
                for name, val in sub_rewards_group.items()
            }
        else: 
            sub_rewards_topk = None

        return (traj_feature_k, noisy_traj_points_k,
                sub_rewards_topk, topk_idx, topk_val)

    def get_vocab_pdm_subscores(self, sub_rewards_group, reward_group, bs, token, dropout_ratio = 0.5,):
        if self.vocab_pdm_score_full is None:
            self.vocab_pdm_score_full = joblib.load(self.vocab_pdm_gt_path)
            self.vocab = nn.Parameter(
                torch.from_numpy(np.load(self.vocab_path)).to(reward_group.device),
                requires_grad=False
            )
        keys = sub_rewards_group[0].keys()
        sub_rewards_group = {
            k: torch.tensor(np.vstack([d[k] for d in sub_rewards_group]),device=reward_group.device,dtype=reward_group.dtype)    # 形状 (B, 1)
            for k in keys
        }
        key_map = {             # 你的内部指标名  ->  vocab 里的字段名
            "no_collision"  : "no_at_fault_collisions",
            "drivable_area" : "drivable_area_compliance",
            "progress"      : "ego_progress",
            "ttc"           : "time_to_collision_within_bound",
            "comfort"       : "history_comfort",
            "dir_weighted"  : "driving_direction_compliance",
        }
        vocab_tensor_dict = {}
        for k_old, k_vocab in key_map.items():
            per_batch = []
            for b in range(bs):
                tok = token[b].item() if torch.is_tensor(token[b]) else token[b]
                arr = self.vocab_pdm_score_full[tok][k_vocab]        # numpy (G_vocab,)
                per_batch.append(
                    torch.as_tensor(arr, device=reward_group.device,
                                    dtype=reward_group.dtype).unsqueeze(0)  # (1, G_vocab)
                )
            vocab_tensor_dict[k_old] = torch.cat(per_batch, dim=0)   # (B, G_vocab)
        final = (
            vocab_tensor_dict["no_collision"]  * vocab_tensor_dict["drivable_area"] *
            (5 * vocab_tensor_dict["ttc"] +        # 权重 5
            5 * vocab_tensor_dict["progress"]  +        # 权重 5
            2 * vocab_tensor_dict["comfort"] )          # 权重 2
        ) / 12                               # 总权重归一
        vocab_tensor_dict["final"] = final
        if "comfort" in vocab_tensor_dict:
            vocab_tensor_dict["comfort"].fill_(-1)

        # -------- 4. 生成随机 dropout 掩码 / 索引 -------------------------------
        G_vocab = final.size(1)
        keep_num = int(G_vocab * (1 - dropout_ratio))                 # 保留个数

        # 保证在同一 device 上采样
        keep_idx = torch.stack([
            torch.randperm(G_vocab, device=reward_group.device)[:keep_num]
            for _ in range(bs)
        ], dim=0)                                                     # (B, keep_num)

        # -------- 5. 用 keep_idx 过滤所有 vocab 张量 ----------------------------
        # helper：按 batch gather
        def batch_gather(t, idx):                                     # t:(B,Gv), idx:(B,K)
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, t.size(-1)//t.size(1)) \
                    if t.dim()==3 else idx
            return t.gather(1, idx_exp)

        for k in vocab_tensor_dict:
            vocab_tensor_dict[k] = batch_gather(vocab_tensor_dict[k], keep_idx)


        for k in keys:  # keys 来自老 dict
            sub_rewards_group[k] = torch.cat(
                [sub_rewards_group[k], vocab_tensor_dict[k]], dim=1
            )
        return sub_rewards_group, keep_idx

    def add_mul_noise(self, diffusion_output, n_aug=3, std_min=0.1, std_max=0.3):
        diffusion_output_aug_list = [diffusion_output]
        for _ in range(n_aug):
            std_dev_t_mul = torch.empty(1, device=diffusion_output.device).uniform_(std_min, std_max).item()
            variance_noise_horizon = randn_tensor(
                [diffusion_output.shape[0],diffusion_output.shape[1],1,1], device=diffusion_output.device, dtype=diffusion_output.dtype
            ) * std_dev_t_mul + 1.0
            variance_noise_vert = randn_tensor(
                [diffusion_output.shape[0],diffusion_output.shape[1],1,1], device=diffusion_output.device, dtype=diffusion_output.dtype
            ) * std_dev_t_mul + 1.0

            variance_noise_mul = torch.cat((variance_noise_horizon,variance_noise_vert),dim=-1)
            variance_noise_mul = variance_noise_mul.repeat(1,1,diffusion_output.shape[2],1)
            diffusion_output_aug_list.append(diffusion_output * variance_noise_mul)
        diffusion_output_aug = torch.cat(diffusion_output_aug_list, dim=1)  # (B, G_all*len(std_dev_t_muls), 8, 3)

        return diffusion_output_aug

    def forward_train_rl(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding,status_feature,camera_feature, targets,global_img, eta,metric_cache,cal_pdm,token) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            step_num = 2
            bs = ego_query.shape[0]
            device = ego_query.device
            self.diffusionrl_scheduler.set_timesteps(1000, device)
            step_ratio = 20 / step_num
            roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
            roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

            num_groups = 2
            # 1. add truncated noise to the plan anchor
            plan_anchor = self.plan_anchor.unsqueeze(0).unsqueeze(0).repeat(bs,num_groups,1,1,1)
            plan_anchor = plan_anchor.view(bs, num_groups * self.ego_fut_mode, *plan_anchor.shape[3:]) # bs num_groups * 20, 8, 2

            diffusion_output = self.norm_odo(plan_anchor)
            noise = torch.randn(diffusion_output.shape, device=device)
            trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
            diffusion_output = self.diffusion_scheduler.add_noise(original_samples=diffusion_output, noise=noise, timesteps=trunc_timesteps)
            all_diffusion_output = [diffusion_output]
            all_log_probs = []
            ego_fut_mode = diffusion_output.shape[1]
            for i, k in enumerate(roll_timesteps[:]):
                # diffusion_output_xy = diffusion_output[..., :2]  # 只保留 x, y
                x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
                noisy_traj_points = self.denorm_odo(x_boxes)

                # 2. proj noisy_traj_points to the query
                traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
                traj_pos_embed = traj_pos_embed.flatten(-2)
                traj_feature = self.plan_anchor_encoder(traj_pos_embed)
                traj_feature = traj_feature.view(bs,ego_fut_mode,-1)

                timesteps = k
                if not torch.is_tensor(timesteps):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    timesteps = torch.tensor([timesteps], dtype=torch.long, device=diffusion_output.device)
                elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                    timesteps = timesteps[None].to(diffusion_output.device)
                
                # 3. embed the timesteps
                timesteps = timesteps.expand(diffusion_output.shape[0])
                time_embed = self.time_mlp(timesteps)
                time_embed = time_embed.view(bs,1,-1)

                # 4. begin the stacked decoder
                poses_reg_list, poses_cls_list,_ = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
                poses_reg = poses_reg_list[-1]
                poses_cls = poses_cls_list[-1]
                x_start = poses_reg[...,:2]
                # x_start = poses_reg
                x_start = self.norm_odo(x_start)
                diffusion_output,log_prob,diffusion_output_mean = self.diffusionrl_scheduler.step(
                    model_output=x_start,
                    timestep=k,
                    sample=diffusion_output,
                    eta=0.0,
                )
            diffusion_output = self.add_mul_noise(diffusion_output,n_aug=2,std_min=0.1,std_max=0.2)
            diffusion_output = self.denorm_odo(diffusion_output)
            diffusion_output = self.bezier_xyyaw(diffusion_output) # B G*N 8 3


            reward_group, metric_cache, sub_rewards_group, sim_traj  = self.get_pdm_score_para(diffusion_output, metric_cache)      # (B,G)
            sub_rewards_group, keep_idx = self.get_vocab_pdm_subscores(sub_rewards_group, reward_group, bs, token, dropout_ratio=0.99)

            vocab = self.vocab[:,::5]
            vocab = vocab.unsqueeze(0).repeat(bs, 1, 1, 1)  # (B,N,8,3)
            vocab = vocab.gather(1, keep_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 3))           # (B, keep_num, 8, 3)
            diffusion_output = torch.cat((diffusion_output, vocab), dim=1)  # (B,G_all,8,3)


        # scorer
        # extract feature
        noisy_traj_points_xy, traj_feature, time_embed = self._get_scorer_inputs(diffusion_output, bs, diffusion_output.shape[1])

        # coarse scorer
        traj_feature_list = self.scorer_decoder(traj_feature, noisy_traj_points_xy, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)        
        traj_feature = traj_feature_list[-1]
        loss_coarse, final_coarse_reward, coarse_reward, sub_loss_dict = self._score_coarse(
            traj_feature, sub_rewards_group,
        )

        # fine scorer
        topk = 32
        traj_feature,noisy_traj_points_xy,sub_rewards_topk,topk_idx,topk_val = self._select_topk(
                                                                                    final_coarse_reward=final_coarse_reward,
                                                                                    topk=topk,
                                                                                    traj_feature=traj_feature,
                                                                                    noisy_traj_points_xy=noisy_traj_points_xy,
                                                                                    sub_rewards_group=sub_rewards_group)
        fine_traj_feature_list = self.fine_scorer_decoder(traj_feature, noisy_traj_points_xy, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)                
        loss_fine, final_fine_reward, fine_reward, fine_sub_loss_dict, fine_reward_dict, fine_best_idx_list = self._score_fine_multi(
                fine_traj_feature_list, sub_rewards_topk,
        )
        sub_loss_dict.update(fine_sub_loss_dict)
        reward_dict = fine_reward_dict
        reward_dict['coarse_reward'] = coarse_reward.mean()
        loss = loss_fine + loss_coarse
        return {"loss":loss,"sub_loss_dict": sub_loss_dict, "reward_dict": reward_dict} 


    def forward_test_rl(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding,status_feature,camera_feature, targets,global_img,metric_cache,eta=1.0,cal_pdm=True,token=None) -> Dict[str, torch.Tensor]:
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusionrl_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        num_groups = 10
        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).unsqueeze(0).repeat(bs,num_groups,1,1,1)
        plan_anchor = plan_anchor.view(bs, num_groups * self.ego_fut_mode, *plan_anchor.shape[3:]) # bs num_groups * 20, 8, 2

        diffusion_output = self.norm_odo(plan_anchor)
        noise = torch.randn(diffusion_output.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        diffusion_output = self.diffusion_scheduler.add_noise(original_samples=diffusion_output, noise=noise, timesteps=trunc_timesteps)
        ego_fut_mode = diffusion_output.shape[1]
        for i, k in enumerate(roll_timesteps[:]):
            x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs,ego_fut_mode,-1)

            timesteps = k
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=diffusion_output.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(diffusion_output.device)
            
            # 3. embed the timesteps
            timesteps = timesteps.expand(diffusion_output.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs,1,-1)

            # 4. begin the stacked decoder
            poses_reg_list, poses_cls_list,_ = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            x_start = poses_reg[...,:2]
            x_start = self.norm_odo(x_start)
            diffusion_output,_,_ = self.diffusionrl_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=diffusion_output,
                eta=0.0,
            )

        diffusion_output = self.add_mul_noise(diffusion_output)
        diffusion_output = self.denorm_odo(diffusion_output)
        diffusion_output = self.bezier_xyyaw(diffusion_output) # B G*N 8 3


        # scorer
        # extract feature
        noisy_traj_points_xy, traj_feature, time_embed = self._get_scorer_inputs(diffusion_output, bs, diffusion_output.shape[1])

        # coarse scorer
        traj_feature_list = self.scorer_decoder(traj_feature, noisy_traj_points_xy, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)        
        traj_feature = traj_feature_list[-1]
        NC_score = self.NC_head(traj_feature).squeeze(-1)             # (B,G)
        EP_score = self.EP_head(traj_feature).squeeze(-1)             # (B,G)
        DAC_score = self.DAC_head(traj_feature).squeeze(-1)             # (B,G)
        TTC_score = self.TTC_head(traj_feature).squeeze(-1)             # (B,G)
        C_score = self.C_head(traj_feature).squeeze(-1)             # (B,G)
        final_coarse_reward = self.sigmoid(NC_score)*self.sigmoid(DAC_score)*(5*self.sigmoid(TTC_score)+5*self.sigmoid(EP_score)+2*self.sigmoid(C_score))/12

        best_coarse_flat = torch.argmax(final_coarse_reward, dim=-1)      # (B,)
        coarse_traj = diffusion_output[
            torch.arange(bs, device=device), best_coarse_flat
        ].unsqueeze(1) 
        traj_to_score = [coarse_traj]

        topk = 32
        traj_feature,noisy_traj_points_xy,sub_rewards_topk,topk_idx,topk_val = self._select_topk(
                                                                                    final_coarse_reward=final_coarse_reward,
                                                                                    topk=topk,
                                                                                    traj_feature=traj_feature,
                                                                                    noisy_traj_points_xy=noisy_traj_points_xy,
                                                                                    sub_rewards_group=None)

        fine_traj_feature_list = self.fine_scorer_decoder(traj_feature, noisy_traj_points_xy, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
        loss_fine, final_fine_reward, fine_reward, fine_sub_loss_dict, fine_reward_dict, fine_best_idx_list = self._score_fine_multi(
                fine_traj_feature_list, sub_rewards_topk,only_reward=True,
        )
        for best_idx_local in fine_best_idx_list:
            global_best_idx = topk_idx[torch.arange(bs, device=device), best_idx_local]
            fine_traj = diffusion_output[
                torch.arange(bs, device=device), global_best_idx
            ].unsqueeze(1)  # (B, 1, 8, 3)
            traj_to_score.append(fine_traj)
            
        traj_to_score = torch.cat(traj_to_score, dim=1)

        # for official eval / inference without PDM scoring
        if not cal_pdm:
            # Get top-32 trajectories for visualization
            topk_trajectories = diffusion_output[
                torch.arange(bs, device=device).unsqueeze(1).expand(-1, topk),
                topk_idx
            ]  # (B, 32, 8, 3)

            return {
                "trajectory": traj_to_score[:,-1],  # Best trajectory (B, 8, 3)
                "trajectory_candidates": traj_to_score,  # All 4 fine-grained candidates (B, 4, 8, 3)
                "trajectory_topk": topk_trajectories,  # Top-32 candidates (B, 32, 8, 3)
                "topk_scores": topk_val,  # Top-32 scores (B, 32)
                "trajectory_coarse": diffusion_output,  # All coarse candidates (B, 200, 8, 3)
                "coarse_scores": final_coarse_reward,  # Coarse scores (B, 200)
            }

        reward_group, metric_cache, sub_rewards_group, _ = self.get_pdm_score_para(traj_to_score, metric_cache)
        reward_dict = {}
        reward_dict['coarse_reward'] = reward_group[:, 0].mean()
        for i in range(len(fine_best_idx_list)):
            reward_dict[f'fine_reward_{i}'] = reward_group[:, i+1].mean()


        bs = traj_to_score.shape[0]
        num_traj_types = traj_to_score.shape[1]

        keys = sub_rewards_group[0].keys()
        
        aggregated_sub_rewards = {
            k: np.vstack([d[k] for d in sub_rewards_group])  # 形状 (B * 4, 1)
            for k in keys
        }
        
        for k, v in aggregated_sub_rewards.items():
            v_tensor = torch.from_numpy(v).squeeze(-1).to(device)
            
            try:
                v_reshaped = v_tensor.reshape(bs, num_traj_types)
            except RuntimeError as e:
                print(f"Error reshaping sub_reward '{k}'. Expected {bs * num_traj_types} elements, but got {v_tensor.numel()}.")
                reward_dict[k] = v_tensor.mean()
                continue
            reward_dict[f'coarse_{k}'] = v_reshaped[:, 0].mean()
            for i in range(3):
                reward_dict[f'fine_{i}_{k}'] = v_reshaped[:, i+1].mean()

        return {'reward_dict': reward_dict}

    def forward_test_rl_temporal_scorer(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        status_feature,
        camera_feature,
        targets,
        global_img,
        metric_cache,
        current_time: float,
        eta=1.0,
        cal_pdm=True,
        token=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with temporal consistency scoring integrated into coarse selection.

        This method combines the learned PDM-style coarse scorer with temporal consistency
        scoring to select trajectories that are both safe/efficient AND consistent with
        previous pseudo-expert trajectory predictions.

        Mathematical formulation:
            S_total(τ) = (1 - λ) * S_pdm(τ) + λ * S_consist(τ)

        where:
            S_pdm = NC * DAC * (5*TTC + 5*EP + 2*C) / 12  (learned scorer)
            S_consist = -Σ_j w_j * ||τ(t_k) - τ_j*(t_k)||_2 / σ  (temporal consistency)
            w_j = α^(k-j-1)  (older predictions get more weight)

        Args:
            ego_query: Ego query features
            agents_query: Agent query features
            bev_feature: BEV feature map
            bev_spatial_shape: Spatial shape of BEV features
            status_encoding: Status encoding
            status_feature: Status feature
            camera_feature: Camera feature
            targets: Training targets (unused in inference)
            global_img: Global image feature (optional)
            metric_cache: PDM metric cache
            current_time: Current simulation time in seconds
            eta: Diffusion eta parameter
            cal_pdm: Whether to calculate PDM scores
            token: Token identifier

        Returns:
            Dictionary containing:
                - trajectory: Best trajectory (B, 8, 3)
                - trajectory_candidates: Fine-grained candidates (B, 4, 8, 3)
                - trajectory_topk: Top-32 candidates (B, 32, 8, 3)
                - topk_scores: Combined scores for top-32 (B, 32)
                - trajectory_coarse: All coarse candidates (B, 200, 8, 3)
                - coarse_scores: Combined coarse scores (B, 200)
                - pdm_coarse_scores: PDM-only coarse scores (B, 200)
                - temporal_coarse_scores: Temporal consistency scores (B, 200)
        """
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusionrl_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        num_groups = 10
        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).unsqueeze(0).repeat(bs, num_groups, 1, 1, 1)
        plan_anchor = plan_anchor.view(bs, num_groups * self.ego_fut_mode, *plan_anchor.shape[3:])

        diffusion_output = self.norm_odo(plan_anchor)
        noise = torch.randn(diffusion_output.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        diffusion_output = self.diffusion_scheduler.add_noise(
            original_samples=diffusion_output, noise=noise, timesteps=trunc_timesteps
        )
        ego_fut_mode = diffusion_output.shape[1]

        # Diffusion denoising loop
        for i, k in enumerate(roll_timesteps[:]):
            x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

            timesteps = k
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=diffusion_output.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(diffusion_output.device)

            timesteps = timesteps.expand(diffusion_output.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs, 1, -1)

            poses_reg_list, poses_cls_list, _ = self.diff_decoder(
                traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding, global_img
            )
            poses_reg = poses_reg_list[-1]
            x_start = poses_reg[..., :2]
            x_start = self.norm_odo(x_start)
            diffusion_output, _, _ = self.diffusionrl_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=diffusion_output,
                eta=0.0,
            )

        # Add multiplicative noise for diversity
        diffusion_output = self.add_mul_noise(diffusion_output)
        diffusion_output = self.denorm_odo(diffusion_output)
        diffusion_output = self.bezier_xyyaw(diffusion_output)  # B G*N 8 3

        # ============ SCORER WITH TEMPORAL CONSISTENCY ============
        noisy_traj_points_xy, traj_feature, time_embed = self._get_scorer_inputs(
            diffusion_output, bs, diffusion_output.shape[1]
        )

        # Coarse scorer - get learned PDM-style scores
        traj_feature_list = self.scorer_decoder(
            traj_feature, noisy_traj_points_xy, bev_feature, bev_spatial_shape,
            agents_query, ego_query, time_embed, status_encoding, global_img
        )
        traj_feature = traj_feature_list[-1]

        NC_score = self.NC_head(traj_feature).squeeze(-1)
        EP_score = self.EP_head(traj_feature).squeeze(-1)
        DAC_score = self.DAC_head(traj_feature).squeeze(-1)
        TTC_score = self.TTC_head(traj_feature).squeeze(-1)
        C_score = self.C_head(traj_feature).squeeze(-1)

        # PDM-style coarse reward (learned scorer)
        pdm_coarse_reward = (
            self.sigmoid(NC_score) * self.sigmoid(DAC_score) *
            (5 * self.sigmoid(TTC_score) + 5 * self.sigmoid(EP_score) + 2 * self.sigmoid(C_score)) / 12
        )  # (B, G)

        # Compute temporal consistency scores (already in [0, 1] range)
        temporal_scores = self.compute_temporal_consistency_score(
            diffusion_output, current_time
        )  # (B, G) in (0, 1]

        # Combined score: weighted sum of PDM and temporal consistency
        # Both scores are now in [0, 1] range, no normalization needed
        final_coarse_reward = (
            (1 - self.temporal_lambda) * pdm_coarse_reward +
            self.temporal_lambda * temporal_scores
        )

        # Select best coarse trajectory
        best_coarse_flat = torch.argmax(final_coarse_reward, dim=-1)
        coarse_traj = diffusion_output[
            torch.arange(bs, device=device), best_coarse_flat
        ].unsqueeze(1)
        traj_to_score = [coarse_traj]

        # Top-k selection for fine scoring
        topk = 32
        traj_feature, noisy_traj_points_xy, sub_rewards_topk, topk_idx, topk_val = self._select_topk(
            final_coarse_reward=final_coarse_reward,
            topk=topk,
            traj_feature=traj_feature,
            noisy_traj_points_xy=noisy_traj_points_xy,
            sub_rewards_group=None
        )

        # Fine scorer
        fine_traj_feature_list = self.fine_scorer_decoder(
            traj_feature, noisy_traj_points_xy, bev_feature, bev_spatial_shape,
            agents_query, ego_query, time_embed, status_encoding, global_img
        )
        loss_fine, final_fine_reward, fine_reward, fine_sub_loss_dict, fine_reward_dict, fine_best_idx_list = self._score_fine_multi(
            fine_traj_feature_list, sub_rewards_topk, only_reward=True,
        )

        for best_idx_local in fine_best_idx_list:
            global_best_idx = topk_idx[torch.arange(bs, device=device), best_idx_local]
            fine_traj = diffusion_output[
                torch.arange(bs, device=device), global_best_idx
            ].unsqueeze(1)
            traj_to_score.append(fine_traj)

        traj_to_score = torch.cat(traj_to_score, dim=1)

        # Get the final selected trajectory (for rollout)
        selected_traj = traj_to_score[:, -1]  # (B, 8, 3)

        # Compute consensus trajectory using PDM scores (NOT combined scores)
        # This avoids circular dependency: temporal consistency should not influence
        # its own reference signal. The consensus represents the model's "confident
        # belief" weighted by pure PDM quality scores.
        consensus_traj = self.compute_consensus_trajectory(
            diffusion_output, pdm_coarse_reward
        )  # (B, 8, 2)

        # Update temporal history with CONSENSUS trajectory (not selected trajectory)
        # Future candidates will be compared against this consensus, encouraging
        # trajectory selection that is consistent with the model's aggregate prediction.
        for b in range(bs):
            self.update_temporal_history(
                consensus_traj[b].cpu().numpy(),  # Store consensus (x, y)
                current_time
            )

        # Return results
        if not cal_pdm:
            topk_trajectories = diffusion_output[
                torch.arange(bs, device=device).unsqueeze(1).expand(-1, topk),
                topk_idx
            ]

            return {
                "trajectory": traj_to_score[:, -1],
                "trajectory_candidates": traj_to_score,
                "trajectory_topk": topk_trajectories,
                "topk_scores": topk_val,
                "trajectory_coarse": diffusion_output,
                "coarse_scores": final_coarse_reward,
                "pdm_coarse_scores": pdm_coarse_reward,
                "temporal_coarse_scores": temporal_scores,
                "temporal_history_length": len(self.temporal_history),
                "consensus_trajectory": consensus_traj,  # Score-weighted mean trajectory
            }

        # If cal_pdm is True, compute actual PDM scores
        reward_group, metric_cache, sub_rewards_group, _ = self.get_pdm_score_para(traj_to_score, metric_cache)
        reward_dict = {}
        reward_dict['coarse_reward'] = reward_group[:, 0].mean()
        for i in range(len(fine_best_idx_list)):
            reward_dict[f'fine_reward_{i}'] = reward_group[:, i + 1].mean()

        bs = traj_to_score.shape[0]
        num_traj_types = traj_to_score.shape[1]
        keys = sub_rewards_group[0].keys()

        aggregated_sub_rewards = {
            k: np.vstack([d[k] for d in sub_rewards_group])
            for k in keys
        }

        for k, v in aggregated_sub_rewards.items():
            v_tensor = torch.from_numpy(v).squeeze(-1).to(device)
            try:
                v_reshaped = v_tensor.reshape(bs, num_traj_types)
            except RuntimeError as e:
                print(f"Error reshaping sub_reward '{k}'. Expected {bs * num_traj_types} elements, got {v_tensor.numel()}.")
                reward_dict[k] = v_tensor.mean()
                continue
            reward_dict[f'coarse_{k}'] = v_reshaped[:, 0].mean()
            for i in range(3):
                reward_dict[f'fine_{i}_{k}'] = v_reshaped[:, i + 1].mean()

        return {'reward_dict': reward_dict}

    def compute_diversity(self, trajectories, eps=1e-6):
        """
        计算 diversity metric。
        
        Args:
            trajectories: Tensor of shape (B, G, T, 3) → 只用 x, y
            eps: small constant to avoid division by zero
        
        Returns:
            diversity: Tensor of shape (B, T) → 每个样本、每个时间步的 diversity 值
        """
        B, G, T, _ = trajectories.shape
        xy = trajectories[..., :2]  # (B, G, T, 2)

        # 计算 pairwise diversity（D_raw）
        diversity_raw = torch.zeros((B, T), device=trajectories.device)
        for i in range(G):
            for j in range(i + 1, G):
                diff = xy[:, i] - xy[:, j]  # (B, T, 2)
                dist = torch.norm(diff, dim=-1)  # (B, T)
                diversity_raw += dist

        diversity_raw = 2 * diversity_raw / (G * (G - 1) + eps)  # (B, T)

        # 计算归一化项：平均模长
        mean_magnitude = torch.norm(xy, dim=-1).mean(dim=1)  # (B, T)

        # 最终归一化结果
        diversity = diversity_raw / (eps + mean_magnitude)  # (B, T)
        diversity = torch.clamp(diversity, max=1.0)  # clip to [0, 1]

        return diversity.mean(dim=1)  # shape (B, T)

    def bezier_xyyaw(self,xy8: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        xy8 : Tensor, shape = (B, G, 8, 2)
            仅包含未来 8 个 (x, y) 预测点，默认以 (0,0) 为局部坐标系原点
        Returns
        -------
        xyyaw : Tensor, shape = (B, G, 8, 3)
                对应 8 个预测点的 (x, y, yaw)（弧度）
        """
        assert xy8.shape[-2:] == (8, 2), "Input must be (B,G,8,2)"
        B, G, _, _ = xy8.shape
        device, dtype = xy8.device, xy8.dtype

        # ---------- ①  在最前面插入固定起点 (0,0) ----------
        origin = torch.zeros_like(xy8[..., :1, :])      # (B,G,1,2)
        ctrl   = torch.cat([origin, xy8], dim=-2)       # (B,G,9,2)
        n      = ctrl.shape[-2] - 1                     # 8 阶 Bézier

        # ΔP_i = P_{i+1} - P_i  → (B,G,8,2)
        delta = ctrl[..., 1:, :] - ctrl[..., :-1, :]

        # 组合数 C(n-1,i),  i = 0…7
        binom = torch.tensor(
            [math.comb(n - 1, i) for i in range(n)],
            device=device, dtype=dtype
        )                                               # (8,)

        # ---------- ②  采样 t_k = k / n ,  k = 1…8 ----------
        t = torch.arange(1, n + 1, device=device, dtype=dtype) / n   # (8,)

        # Bernstein 基函数 (一阶导数用)  → (8,8)
        t_pow   = t.view(-1, 1) ** torch.arange(0, n,     device=device, dtype=dtype)
        one_pow = (1 - t).view(-1, 1) ** torch.arange(n-1, -1, -1, device=device, dtype=dtype)
        basis   = binom * t_pow * one_pow

        # 扩维广播
        delta_exp = delta.unsqueeze(2)                  # (B,G,1,8,2)
        basis_exp = basis.view(1, 1, 8, 8, 1)           # (1,1,8,8,1)

        # 一阶导：B'(t_k) = n * Σ_i basis_i(t_k) * ΔP_i
        deriv = n * (delta_exp * basis_exp).sum(dim=3)  # (B,G,8,2)

        # yaw = atan2(dy, dx)
        dx, dy = deriv[..., 0], deriv[..., 1]
        yaw = torch.atan2(dy, dx).unsqueeze(-1)         # (B,G,8,1)

        return torch.cat([xy8, yaw], dim=-1)            # (B,G,8,3)