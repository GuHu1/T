import copy
from math import pi, cos, sin
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss 
from mmdet.models.dense_heads import DETRHead
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner import BaseModule
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn import Linear, bias_init_with_prob
from torch.cuda.amp.autocast_mode import autocast
from mmdet.models.backbones.resnet import BasicBlock

from .tokenlearner import *
from .rcsample import Mlp, SELayer
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding

class MLN(nn.Module):
    ''' 
    from "https://github.com/exiawsh/StreamPETR"
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out

class SELayerMLP(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.mlp_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.mlp_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.mlp_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.mlp_expand(x_se)
        return x * self.gate(x_se)


class FactorSelfAttention(nn.Module):
    """Shared one-step transition operator acting inside a single factor subspace.

    Input/output are ``[num_tokens, batch, rank]`` (sequence-first, matching the
    convention of the other transformer layers in this file). Self-attention runs
    over the token axis, i.e. over the scene queries. The same weights are reused
    for every factor direction and for both the main and the auxiliary path, so the
    module learns one latent transition operator instead of a per-direction or
    per-timestep predictor.
    """
    def __init__(self, rank, num_heads=4, ffn_ratio=2.0, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(rank, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(rank)
        self.norm2 = nn.LayerNorm(rank)
        hidden = int(rank * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(rank, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, rank))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class OrthogonalPredictiveFactorization(nn.Module):
    """Orthogonal predictive factorization (OPF) over temporal dynamic residuals.

    Given the current and previous scene-query residuals ``R_t`` and ``R_{t-1}``
    (both ``[B, N, dim]``), the residual space is partitioned into ``num_factors``
    orthogonal subspaces through learned projectors ``P_k in R^{dim x rank}`` with
    ``num_factors * rank == dim``:

    * each factor is projected as ``R^{(k)} = P_k^T R``;
    * a shared self-attention operator plays the role of the predictor, giving the
      auxiliary path ``C_k = SA(R_{t-1}^{(k)})`` and the main path ``T_k = SA(R_t^{(k)})``;
    * the predicted dynamic state is re-synthesized as ``R_hat = sum_k P_k T_k``.

    Stop-gradient follows the JEPA convention on the target branch only: ``r_cur`` is
    detached when its projection serves as the regression target of ``L_aux``. The
    context branch (``r_prev``) stays online, so ``L_aux`` also back-propagates into
    the previous-frame scene queries, and the main path back-propagates the
    downstream planning loss through ``r_hat`` into the projectors and ``r_cur``.
    The shared SA is trained by ``L_aux`` and by the main path; the projectors are
    trained by the auxiliary, orthogonality and factor-activity terms (plus the main
    path), matching the JEPA-Anything formulation in which "the projectors remain
    trainable and receive gradients from the predictive, orthogonality, and
    factor-activity terms".
    """
    def __init__(self, dim=256, num_factors=4, num_heads=4, ffn_ratio=2.0):
        super().__init__()
        assert dim % num_factors == 0, \
            'dim ({}) must be divisible by num_factors ({})'.format(dim, num_factors)
        self.dim = dim
        self.num_factors = num_factors
        self.rank = dim // num_factors
        assert self.rank % num_heads == 0, \
            'rank ({}) must be divisible by num_heads ({})'.format(self.rank, num_heads)

        # Joint orthogonal initialization: the stacked projector [P_1, ..., P_K] is
        # drawn as a single dim x dim orthogonal matrix, so the K factor subspaces
        # are mutually orthogonal (P_i^T P_j = 0) and every block is orthonormal
        # (P_k^T P_k = I_rank) at initialization. Hence L_orth == 0 and
        # sum_k P_k P_k^T == I at init: the OPF synthesis starts as an exact identity
        # decomposition of the residual space, instead of K overlapping random bases.
        stacked = torch.empty(dim, dim)
        nn.init.orthogonal_(stacked)
        self.projections = nn.ParameterList(
            nn.Parameter(stacked[:, k * self.rank:(k + 1) * self.rank].contiguous())
            for k in range(num_factors))
        self.sa = FactorSelfAttention(self.rank, num_heads=num_heads, ffn_ratio=ffn_ratio)

    def forward(self, r_cur, r_prev):
        # r_cur, r_prev: [B, N, dim]
        r_target = r_cur.detach()          # stop-gradient target (JEPA convention)
        z_target, c_pred, t_pred = [], [], []
        for P in self.projections:
            z_target.append(r_target @ P)  # R_t^{(k)} = P_k^T R_t  -> [B, N, rank]
            z_prev = r_prev @ P            # R_{t-1}^{(k)}
            c_pred.append(self.sa(z_prev.permute(1, 0, 2)).permute(1, 0, 2))
            t_pred.append(self.sa((r_cur @ P).permute(1, 0, 2)).permute(1, 0, 2))

        r_hat = sum(t @ P.t() for t, P in zip(t_pred, self.projections))  # [B, N, dim]
        return r_hat, z_target, c_pred


@HEADS.register_module()
class ResWorldHead(BaseModule):
    def __init__(self,
                #  *args,
                 grid_config,
                 num_frames=3,
                 embed_dims=256,
                 in_channels=256,
                 num_reg_fcs=2,
                 positional_encoding=None,
                 bev_h=30,
                 bev_w=30,
                 fut_ts=6,
                 fut_mode=6,
                 num_scenes=16,
                 latent_decoder=None,
                 res_latent_decoder=None,
                 way_decoder=None,
                 num_factors=4,
                 opf_num_heads=4,
                 opf_ffn_ratio=2.0,
                 lambda_opf_aux=1.0,
                 lambda_opf_orth=1.0,
                 lambda_opf_fac=1.0,
                 gamma_fac=0.1,
                 gamma_fac_ratio=0.0,
                 ego_fut_mode=3,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                 ego_lcf_feat_idx=None,
                 valid_fut_ts=6,
                 **kwargs):
        super(ResWorldHead, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.embed_dims = embed_dims
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.latent_decoder = latent_decoder
        self.res_latent_decoder = res_latent_decoder
        self.way_decoder = way_decoder
        self.positional_encoding = positional_encoding
        self.ego_fut_mode = ego_fut_mode
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.valid_fut_ts = valid_fut_ts
        self.num_scenes = num_scenes
        self.num_frames = num_frames
        self.num_factors = num_factors
        self.opf_num_heads = opf_num_heads
        self.opf_ffn_ratio = opf_ffn_ratio
        self.lambda_opf_aux = lambda_opf_aux
        self.lambda_opf_orth = lambda_opf_orth
        self.lambda_opf_fac = lambda_opf_fac
        self.gamma_fac = gamma_fac
        self.gamma_fac_ratio = gamma_fac_ratio
        self.grid_min = torch.tensor([grid_config['x'][0], grid_config['y'][0]])
        self.grid_max = torch.tensor([grid_config['x'][1], grid_config['y'][1]])
        self.grid_size = torch.tensor([grid_config['x'][2], grid_config['y'][2]])

        self._init_layers()
        self.loss_plan_reg = build_loss(loss_plan_reg)
        self.loss_plan_reg_init = build_loss(loss_plan_reg)
                
    def _init_layers(self):
        ego_fut_decoder = []
        ego_fut_dec_in_dim = self.embed_dims + len(self.ego_lcf_feat_idx) \
            if self.ego_lcf_feat_idx is not None else self.embed_dims
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)
        init_ego_fut_decoder = []
        for _ in range(self.num_reg_fcs):
            init_ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            init_ego_fut_decoder.append(nn.ReLU())
        init_ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, 2))
        self.init_ego_fut_decoder = nn.Sequential(*init_ego_fut_decoder)

        self.navi_embedding = nn.Embedding(3, self.embed_dims)
        self.navi_se = SELayerMLP(self.embed_dims)
        self.canbus_mlp = Mlp(18, self.embed_dims, self.embed_dims)
        self.canbus_se = SELayerMLP(self.embed_dims)
        self.bev_fusion_conv = ConvModule(self.in_channels * self.num_frames, self.in_channels, 
                                          kernel_size=3, padding=1)

        self.way_point = nn.Embedding(self.ego_fut_mode*self.fut_ts, self.embed_dims * 2)
        self.tokenlearner = TokenLearner(self.num_scenes, self.embed_dims * 2)
        self.res_tokenlearner = TokenLearner(self.num_scenes, self.embed_dims * 2)
        self.tokenfuser = TokenFuser(self.num_scenes, 256)

        self.latent_decoder = build_transformer_layer_sequence(self.latent_decoder)
        self.way_decoder = build_transformer_layer_sequence(self.way_decoder)
        self.res_latent_decoder = build_transformer_layer_sequence(self.res_latent_decoder)
        self.opf = OrthogonalPredictiveFactorization(
            self.embed_dims, num_factors=self.num_factors,
            num_heads=self.opf_num_heads, ffn_ratio=self.opf_ffn_ratio)
        self.col_attn = MultiScaleDeformableAttention(self.embed_dims,
                                            num_points=8, num_levels=1)
        self.action_mln = MLN(6*2)
        self.positional_encoding = build_positional_encoding(
            self.positional_encoding)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        if self.latent_decoder is not None:
            for p in self.latent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p) 
        if self.way_decoder is not None:
            for p in self.way_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.res_latent_decoder is not None:
            for p in self.res_latent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    @force_fp32(apply_to=('bev_feats'))
    def forward(self,
                bev_inputs,
                img_metas,
                prev_bev=None,
                only_bev=False,
                ego_his_trajs=None,
                ego_lcf_feat=None,
                cmd=None
            ):
        
        bev_feats, can_bus_infos = bev_inputs
        bt, c, h, w = bev_feats.shape
        bs = bt // self.num_frames
        dtype = bev_feats[0].dtype
        device = bev_feats[0].device
        can_bus_infos = self.canbus_mlp(can_bus_infos.permute(1, 0, 2)).view(bt, 1, self.in_channels)
        bev_embed = self.canbus_se(bev_feats.view(bt, c, h*w).permute(0, 2, 1), can_bus_infos)
        bev_embed_single = bev_embed.clone()
        bev_embed = bev_embed.permute(0, 2, 1).view(self.num_frames, bs, c, h, w).permute(1, 0, 2, 3, 4)
        bev_embed = self.bev_fusion_conv(bev_embed.reshape(bs, self.num_frames * c, h, w))
        bev_feats = bev_feats.view(self.num_frames, bs, c, h, w)


        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_feats.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        pos_embd = bev_pos.flatten(2).permute(0, 2, 1)
        bev_embed = bev_embed.reshape(bs, c, h * w).permute(0, 2, 1)
        # res_embed = res_embed.reshape(bs, c, h * w).permute(0, 2, 1)

        navi_embed = []
        for bidx in range(bs):
            cmd_idx = torch.nonzero(cmd[bidx, 0, 0])[0, 0]
            navi_embed.append(self.navi_embedding.weight[cmd_idx][None, None])
        navi_embed = torch.cat(navi_embed, dim=0)

        bev_navi_embed = self.navi_se(bev_embed, navi_embed)

        bev_query = torch.cat((bev_navi_embed, pos_embd), -1)

        learned_latent_query, selected = self.tokenlearner(bev_query)
        _, res_selected = self.res_tokenlearner(bev_query)
        bev_embed_single = torch.cat((bev_embed_single, pos_embd.repeat(self.num_frames,1,1)), -1)
        bev_embed_single = torch.einsum('bsi,bic->bsc', res_selected.repeat(self.num_frames,1,1), bev_embed_single)\
                            .view(self.num_frames, bs, learned_latent_query.shape[1],learned_latent_query.shape[2])
        # Per-frame scene queries S_t, S_{t-1}, S_{t-2} (newest first). The positional
        # half of the concatenated query is identical across frames, so only the
        # content half is carried into the temporal residual.
        scene_queries = bev_embed_single[..., :self.embed_dims]
        assert scene_queries.shape[0] >= 3, \
            'OPF needs at least 3 frames (R_t and R_{t-1}), got {}'.format(scene_queries.shape[0])
        res_latent_query_all = scene_queries[:-1] - scene_queries[1:]  # [R_t, R_{t-1}]

        learned_latent_query=learned_latent_query.permute(1, 0, 2)
        latent_query, latent_pos = torch.split(
            learned_latent_query, self.embed_dims, dim=2)

        latent_query = self.latent_decoder(
                query=latent_query,
                key=latent_query,
                value=latent_query,
                query_pos=latent_pos,
                key_pos=latent_pos)

        way_point = self.way_point.weight.to(dtype)
        wp_pos, way_point = torch.split(
            way_point, self.embed_dims, dim=1)

        wp_pos = wp_pos.unsqueeze(0).expand(bs, -1, -1)
        way_point = way_point.unsqueeze(0).expand(bs, -1, -1)
        wp_pos = wp_pos.permute(1, 0, 2)
        way_point = way_point.permute(1, 0, 2)

        way_point = self.way_decoder(
                query=way_point,
                key=latent_query,
                value=latent_query,
                query_pos=wp_pos,
                key_pos=latent_pos)
        init_ego_trajs = self.init_ego_fut_decoder(way_point)
        init_ego_trajs = init_ego_trajs.permute(1, 0, 2). view(bs, 
                                                    self.ego_fut_mode, self.fut_ts, 2)
        init_ego_coords = init_ego_trajs.cumsum(dim=2).view(bs, -1, 2)
        init_ego_coords = (init_ego_coords - self.grid_min.to(device).view(1, 1, 2)) / \
                            self.grid_size.to(device).view(1, 1, 2) / 200
        init_wp_vector = []
        for bidx in range(bs):
            cmd_idx = torch.nonzero(cmd[bidx, 0, 0])[0, 0]
            init_wp_vector.append(init_ego_trajs[bidx, cmd_idx, ...].reshape(1, 1, 12))  
        init_wp_vector = torch.cat(init_wp_vector, dim=1)

        reference_points = init_ego_coords.unsqueeze(2)
        spatial_shapes = torch.tensor([self.bev_w, self.bev_h]).view(1, 2).to(device)
        level_start_index = torch.tensor([0]).to(device)

        # OPF: factorize the current / previous temporal residuals into K orthogonal
        # directions, predict each direction with a shared self-attention operator and
        # synthesize the predicted dynamic state R_hat, which replaces the former
        # monolithic dynamic-object representation.
        r_cur = res_latent_query_all[0]                                    # R_t
        r_prev = res_latent_query_all[1]                                   # R_{t-1}
        r_hat, opf_z_target, opf_c_pred = self.opf(r_cur, r_prev)          # [bs, N, C]
        res_latent_query = self.action_mln(r_hat.permute(1, 0, 2), init_wp_vector)
        # The positional half of the residual is identically zero (the positional
        # embedding is frame-invariant), so query_pos / key_pos stay zero tensors,
        # exactly as in the original residual path.
        res_latent_pos = torch.zeros_like(res_latent_query)
        res_latent_query = self.res_latent_decoder(
                query=res_latent_query,
                key=res_latent_query,
                value=res_latent_query,
                query_pos=res_latent_pos,
                key_pos=res_latent_pos)

        pred_bev = self.tokenfuser(res_latent_query.permute(1, 0, 2), bev_navi_embed) + bev_navi_embed

        way_point = self.col_attn(
                query=way_point,
                key=pred_bev.permute(1, 0, 2),
                value=pred_bev.permute(1, 0, 2),
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index) 

        outputs_ego_trajs = self.ego_fut_decoder(way_point)
        outputs_ego_trajs = outputs_ego_trajs.permute(1, 0, 2). view(bs, 
                                                      self.ego_fut_mode, self.fut_ts, 2)

        wp_vector = []
        for bidx in range(bs):
            cmd_idx = torch.nonzero(cmd[bidx, 0, 0])[0, 0]
            wp_vector.append(outputs_ego_trajs[bidx, cmd_idx, ...].reshape(1, 1, 12))  
        wp_vector = torch.cat(wp_vector, dim=1)

        outs = {
            'bev_embed': bev_embed,
            'pred_bev': pred_bev,
            'scene_query': latent_query,
            'wp_vector': wp_vector,
            # 'act_query': act_query,
            # 'act_pos': act_pos,
            'ego_fut_preds': outputs_ego_trajs,
            # 'ego_fut_preds': init_ego_trajs,
            'init_ego_fut_preds': init_ego_trajs,
            'opf_z_target': opf_z_target,
            'opf_c_pred': opf_c_pred,
        }

        return outs
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             ego_fut_gt,
             ego_fut_masks,
             ego_fut_cmd,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None):

        ego_fut_preds = preds_dicts['ego_fut_preds']
        init_ego_fut_preds = preds_dicts['init_ego_fut_preds']
        loss_dict = dict()

        # Planning Loss
        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )

        loss_plan_l1_init = self.loss_plan_reg_init(
            init_ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )
      
        loss_dict['loss_plan_reg'] = loss_plan_l1
        loss_dict['loss_plan_reg_init'] = loss_plan_l1_init

        # OPF terms: auxiliary dynamic prediction + projectors regularization.
        loss_aux, loss_orth, loss_fac, opf_stats = self.opf_losses(preds_dicts)
        loss_dict['loss_opf_aux'] = self.lambda_opf_aux * loss_aux
        loss_dict['loss_opf_orth'] = self.lambda_opf_orth * loss_orth
        loss_dict['loss_opf_fac'] = self.lambda_opf_fac * loss_fac
        # Detached diagnostics for the logger. Their keys carry no 'loss' substring, so
        # mmdet's _parse_losses logs them without adding them to the objective.
        loss_dict.update(opf_stats)

        return loss_dict

    def resolve_gamma(self, z_target):
        """Activity floor gamma_fac used by L_fac.

        ``gamma_fac`` is an absolute floor and must be calibrated to the sigma scale,
        but that scale is not constant: measured on nuScenes it grows by ~4x (median)
        to ~16x (minimum) over the first four epochs, so a fixed absolute gamma
        progressively stops firing as the representation grows.

        * ``gamma_fac_ratio <= 0`` (default): the absolute ``gamma_fac`` is used, i.e.
          the paper's fixed floor.
        * ``gamma_fac_ratio > 0``: the floor follows the representation scale,
          ``gamma = max(gamma_fac, gamma_fac_ratio * median(sigma))``, so the fraction
          of coordinates that L_fac acts on stays put across epochs instead of
          vanishing. ``gamma_fac`` then only acts as a never-vanish floor.

        The median is detached, so gamma is a per-step constant: the term keeps its
        single gradient path (through sigma inside the hinge) and no gradient flows
        back through the threshold.
        """
        floor = torch.as_tensor(
            self.gamma_fac, device=z_target[0].device, dtype=torch.float32)
        if self.gamma_fac_ratio <= 0:
            return floor
        with torch.no_grad():
            sigma = torch.cat([z.reshape(-1, self.opf.rank).std(dim=0).float()
                               for z in z_target])
            return torch.max(floor, self.gamma_fac_ratio * sigma.median())

    def opf_losses(self, preds_dicts):
        """Auxiliary-prediction, orthogonality and factor-activity terms of OPF.

        The auxiliary term supervises the shared self-attention operator on the
        observed pair (R_{t-1}^{(k)} -> R_t^{(k)}); the orthogonality term shapes the
        projectors so the K directions stay non-redundant; the factor-activity term
        keeps every projected coordinate alive so the directions cannot all collapse
        to constants.
        """
        z_target = preds_dicts['opf_z_target']
        c_pred = preds_dicts['opf_c_pred']
        projections = self.opf.projections
        num_factors = self.opf.num_factors
        rank = self.opf.rank
        ref = z_target[0]

        # L_aux = mean_k MSE(C_k, R_t^{(k)}), with R_t^{(k)} already stop-gradient.
        loss_aux = ref.new_zeros(())
        for z_k, c_k in zip(z_target, c_pred):
            loss_aux = loss_aux + F.mse_loss(c_k, z_k)
        loss_aux = loss_aux / num_factors

        # L_orth = sum_k ||P_k^T P_k - I_r||_F^2 + sum_{i<j} ||P_i^T P_j||_F^2.
        eye = torch.eye(rank, device=ref.device, dtype=ref.dtype)
        loss_orth = ref.new_zeros(())
        for P in projections:
            loss_orth = loss_orth + (P.t() @ P - eye).pow(2).sum()
        for i in range(num_factors):
            for j in range(i + 1, num_factors):
                loss_orth = loss_orth + (projections[i].t() @ projections[j]).pow(2).sum()

        # L_fac = mean_{k,j} max(0, gamma - sigma_{k,j}), with gamma resolved by
        # resolve_gamma (the absolute gamma_fac, or a fixed ratio of the current
        # median sigma). torch.std is used directly: its backward is safe at zero
        # variance (returns 0, verified) and gives a scale-free revival force for
        # nearly collapsed coordinates, which the paper's sqrt(Var + eps) form would
        # damp by the same eps.
        gamma = self.resolve_gamma(z_target).to(ref.dtype)
        loss_fac = ref.new_zeros(())
        for z_k in z_target:
            sigma = z_k.reshape(-1, rank).std(dim=0)
            loss_fac = loss_fac + F.relu(gamma - sigma).sum()
        loss_fac = loss_fac / (num_factors * rank)

        stats = self.opf_stats(z_target, loss_aux, loss_orth, loss_fac)
        return loss_aux, loss_orth, loss_fac, stats

    def opf_stats(self, z_target, loss_aux, loss_orth, loss_fac):
        """Detached OPF diagnostics, reported through ``loss_dict`` for logging only.

        mmdet's ``_parse_losses`` adds a term to the optimised objective only when its
        key contains ``'loss'``, so the ``opf_*`` entries below reach the text log and
        TensorBoard without ever contributing to the loss or to backward.

        Reported quantities:
        * ``opf_aux_raw`` / ``opf_orth_raw`` / ``opf_fac_raw`` -- the unweighted OPF
          terms (the ``lambda_*`` factors are applied only to the ``loss_opf_*``
          entries), so the three terms can be compared with each other and with the
          base plan losses on one scale;
        * ``opf_sigma_min`` / ``opf_sigma_p05`` / ``opf_sigma_p25`` / ``opf_sigma_med``
          -- the distribution of per-coordinate factor activity over the
          ``num_factors * rank == dim`` projected coordinates. A single coordinate is
          estimated from only ``bs * num_scenes`` rows (32 with samples_per_gpu=2), but
          these quantiles aggregate 256 coordinates and the median is stable to ~1%;
        * ``opf_gamma`` -- the activity floor actually used this step, i.e. the value
          returned by ``resolve_gamma`` (equal to ``gamma_fac`` in the default
          absolute mode, or ``gamma_fac_ratio * median(sigma)`` in ratio mode). It is
          needed to read ``opf_act_rate`` when the floor is scale-following;
        * ``opf_act_rate`` -- fraction of coordinates whose sigma falls below
          ``gamma``, i.e. whether L_fac fires at all. While it is 0 the term is
          inert; as it approaches 1 the term starts dragging the projectors off the
          orthogonal partition and fighting L_orth;
        * ``opf_ident_err`` -- ``||sum_k P_k P_k^T - I||_F / sqrt(dim)``, the deviation
          of the projectors from the orthogonal partition of the residual space. It is
          ~0 under the joint-orthogonal initialization and should stay below ~0.05.
        """
        with torch.no_grad():
            rank = self.opf.rank
            dim = self.opf.dim
            ref = z_target[0]
            dtype = ref.dtype

            sigma = torch.cat(
                [z.reshape(-1, rank).std(dim=0) for z in z_target]).float()
            gamma = self.resolve_gamma(z_target).to(sigma.dtype)
            act_rate = (sigma < gamma).to(sigma.dtype).mean()
            quant = torch.quantile(
                sigma, sigma.new_tensor([0.05, 0.25, 0.5]))

            stacked = torch.cat([p.detach().float() for p in self.opf.projections],
                                dim=1)
            eye = torch.eye(dim, device=stacked.device, dtype=stacked.dtype)
            ident_err = (stacked @ stacked.t() - eye).norm() / dim ** 0.5

            return {
                'opf_aux_raw': loss_aux.detach(),
                'opf_orth_raw': loss_orth.detach(),
                'opf_fac_raw': loss_fac.detach(),
                'opf_sigma_min': sigma.min().to(dtype),
                'opf_sigma_p05': quant[0].to(dtype),
                'opf_sigma_p25': quant[1].to(dtype),
                'opf_sigma_med': quant[2].to(dtype),
                'opf_gamma': gamma.to(dtype),
                'opf_act_rate': act_rate.to(dtype),
                'opf_ident_err': ident_err.to(dtype),
            }


