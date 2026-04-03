import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .vit_fpn import ViTFeaturePyramid
from .mv_transformer import MultiViewFeatureTransformer, batch_features_camera_parameters
from .matching import warp_indices_with_pose_depth
from .utils import mv_feature_add_position
from .ldm_unet.unet import UNetModel, AttentionBlock
from einops import rearrange

import os
import numpy as np
from .depth_anything.dpt import DepthAnything

from torch.autograd import Function
from torch.autograd.function import once_differentiable
import smm_cuda



# -------------------------------------------------------------------------
# Sparse Matrix Multiplication (SMM)
# -------------------------------------------------------------------------
class SMM_QmK(Function):
    @staticmethod
    def forward(ctx, A, B, index):
        ctx.save_for_backward(A, B, index)
        A = A.float()
        B = B.float()
        return smm_cuda.SMM_QmK_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        A, B, index = ctx.saved_tensors
        A = A.float()
        B = B.float()
        grad_output = grad_output.float()
        grad_A, grad_B = smm_cuda.SMM_QmK_backward_cuda(
            grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
        )
        return grad_A, grad_B, None

class SMM_AmV(Function):
    @staticmethod
    def forward(ctx, A, B, index):
        ctx.save_for_backward(A, B, index)
        A = A.float()
        B = B.float()
        return smm_cuda.SMM_AmV_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        A, B, index = ctx.saved_tensors
        A = A.float()
        B = B.float()
        grad_output = grad_output.float()
        grad_A, grad_B = smm_cuda.SMM_AmV_backward_cuda(
            grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
        )
        return grad_A, grad_B, None




# -------------------------------------------------------------------------
# Gaussian Focused Module (GFM)
# -------------------------------------------------------------------------
class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, layer_id, window_size, num_heads, num_topk, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.window_size = window_size
        self.num_heads = num_heads
        self.num_topk = num_topk
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.eps = 1e-10

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.num_heads))
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.topk = self.num_topk[self.layer_id]

    def forward(self, qkvp, pfa_values, pfa_indices, rpi, mask=None, shift=0):
        b_, n, c4 = qkvp.shape
        c = c4 // 4
        qkvp = qkvp.reshape(b_, n, 4, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v, v_lepe = qkvp[0], qkvp[1], qkvp[2], qkvp[3]

        q = q * self.scale
        if pfa_indices[shift] is None:
            current_topk = self.window_size[0] * self.window_size[1]
            attn = (q @ k.transpose(-2, -1))
            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            if not self.training:
                attn.add_(relative_position_bias)
            else:
                attn = attn + relative_position_bias

            if shift:
                nw = mask.shape[0]
                attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, n, n)
        else:
            current_topk = pfa_indices[shift].shape[-1]
            q = q.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            k = k.contiguous().view(b_ * self.num_heads, n, c // self.num_heads).transpose(-2, -1)
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, current_topk).int()
            attn = SMM_QmK.apply(q, k, smm_index).view(b_, self.num_heads, n, current_topk)

            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0).expand(b_, self.num_heads, n, n)
            relative_position_bias = torch.gather(relative_position_bias, dim=-1, index=pfa_indices[shift])
            if not self.training:
                attn.add_(relative_position_bias)
            else:
                attn = attn + relative_position_bias

        if not self.training:
            attn = torch.softmax(attn, dim=-1, out=attn)
        else:
            attn = self.softmax(attn)

        if pfa_values[shift] is not None:
            if not self.training:
                attn.mul_(pfa_values[shift])
                attn.add_(self.eps)
                denom = attn.sum(dim=-1, keepdim=True).add_(self.eps)
                attn.div_(denom)
            else:
                attn = (attn * pfa_values[shift])
                attn = (attn + self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)

        if self.topk < current_topk:
            current_topk = self.topk
            topk_values, topk_indices = torch.topk(attn, current_topk, dim=-1, largest=True, sorted=False)
            attn = topk_values
            if pfa_indices[shift] is not None:
                pfa_indices[shift] = torch.gather(pfa_indices[shift], dim=-1, index=topk_indices)
            else:
                pfa_indices[shift] = topk_indices

        pfa_values[shift] = attn

        if pfa_indices[shift] is None:
            x = ((attn @ v) + v_lepe).transpose(1, 2).reshape(b_, n, c)
        else:
            attn = attn.view(b_ * self.num_heads, n, current_topk)
            v = v.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, current_topk).int()
            x = (SMM_AmV.apply(attn, v, smm_index).view(b_, self.num_heads, n, c // self.num_heads) + v_lepe).transpose(1, 2).reshape(b_, n, c)

        x = self.proj(x)
        return x, pfa_values, pfa_indices

class GaussianFocusedLayer(nn.Module):
    def __init__(self, dim, window_size, num_topk, shift_size,
                 convffn_kernel_size=5, mlp_ratio=4, layer_id=0, num_heads=1, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias

        self.mv_norm1 = nn.LayerNorm(dim)
        self.mv_norm2 = nn.LayerNorm(dim)

        self.w_Wq = nn.Linear(dim, dim)
        self.w_Wk = nn.Linear(dim, dim)
        self.w_Wv = nn.Linear(dim, dim)

        self.convlepe_kernel_size = convffn_kernel_size
        self.v_LePE = dwconv(hidden_features=dim, kernel_size=self.convlepe_kernel_size)

        self.attn_win = WindowAttention(
            self.dim,
            layer_id=layer_id,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            num_topk=num_topk,
            qkv_bias=qkv_bias, )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, act_layer=nn.GELU)

        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features_mv, w_pfa_values, w_pfa_indices, params):
        bv, c, h, w = features_mv.shape
        x_size = (h, w)

        c4 = 4 * c

        features_mv = rearrange(features_mv, "bv c h w -> bv (h w) c", bv=bv, h=h, w=w, c=c)
        x_shortcut = features_mv.clone()
        features_mv = self.mv_norm1(features_mv)

        w_q = self.w_Wq(features_mv)
        w_k = self.w_Wk(features_mv)
        w_v = self.w_Wv(features_mv)

        v_lepe = self.v_LePE(w_v, x_size)
        x_qkvp = torch.cat([w_q, w_k, w_v, v_lepe], dim=-1)

        if self.shift_size > 0:
            shift = 1
            shifted_x = torch.roll(x_qkvp.reshape(bv, h, w, c4), shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shift = 0
            shifted_x = x_qkvp.reshape(bv, h, w, c4)

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c4)
        attn_windows, w_pfa_values, w_pfa_indices = self.attn_win(x_windows, w_pfa_values, w_pfa_indices, rpi=params['rpi_sa'], mask=params['attn_mask'], shift=shift)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            x_win = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_win = shifted_x

        x_win = rearrange(x_win, "bv h w c -> bv (h w) c", bv=bv, h=h, w=w, c=c)
        x = x_shortcut + x_win
        x = x + self.convffn(self.mv_norm2(x), x_size)
        x = rearrange(x, "bv (h w) c -> bv c h w", bv=bv, h=h, w=w, c=c)

        return x, w_pfa_values, w_pfa_indices





# -------------------------------------------------------------------------
# Warp-Index Epipolar Attention (WIEA)
# -------------------------------------------------------------------------
class WarpIndexEpipolarAttention(nn.Module):
    def __init__(self, dim, num_depth_candidates, max_uf, upsample_factor, mlp_ratio, convffn_kernel_size, grid_sample_disable_cudnn=False, num_heads=1, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.upsample_factor = upsample_factor
        self.num_depth_candidates = num_depth_candidates
        self.grid_sample_disable_cudnn = grid_sample_disable_cudnn

        # Positional encoding has a significant impact on the performance
        self.k_pe = nn.Parameter(torch.zeros(1, 1, num_depth_candidates))
        self.v_pe = nn.Parameter(torch.zeros(1, 1, num_depth_candidates))
        self.rpe = nn.Parameter(torch.zeros(1, 1, num_depth_candidates))
        nn.init.trunc_normal_(self.k_pe, std=.02)
        nn.init.trunc_normal_(self.v_pe, std=.02)
        nn.init.trunc_normal_(self.rpe, std=.02)

        self.mv_norm1 = nn.LayerNorm(dim)
        self.mv_norm2 = nn.LayerNorm(dim)
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim, dim)
        self.eps = 1e-10

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, act_layer=nn.GELU)

        if max_uf / upsample_factor == 1:
            unet_channels = 32
            unet_channel_mult = [1, 1, 1]
            unet_num_res_blocks = 1
            unet_attn_resolutions = [4]
        elif max_uf / upsample_factor == 2:
            unet_channels = 16
            unet_channel_mult = [1, 1, 1, 1]
            unet_num_res_blocks = 1
            unet_attn_resolutions = [8]
        else:
            unet_channels = 8
            unet_channel_mult = [1, 1, 1, 1, 1]
            unet_num_res_blocks = 1
            unet_attn_resolutions = [16]

        self.Sim_UNet = nn.Sequential(nn.Conv2d(num_depth_candidates + 1, unet_channels, 3, 1, 1),
                                      nn.GroupNorm(8, unet_channels),
                                      nn.GELU(),
                                      UNetModel(image_size=None, in_channels=unet_channels, model_channels=unet_channels, out_channels=unet_channels,
                                                num_res_blocks=unet_num_res_blocks, attention_resolutions=unet_attn_resolutions,
                                                channel_mult=unet_channel_mult, num_head_channels=unet_channels // 4, dims=2, postnorm=False,
                                                num_frames=2, use_cross_view_self_attn=True, ),
                                      nn.Conv2d(unet_channels, num_depth_candidates, 3, 1, 1), )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features_mv, depth_in, warp_indices, depth_prob, depth_candidates, extrinsics, intrinsics, nn_matrix):
        b, v, c, h, w = features_mv.shape
        x_size = (h, w)
        d = self.num_depth_candidates

        features_mv = rearrange(features_mv, "b v c h w -> (b v) (h w) c", b=b, v=v, h=h, w=w, c=c)
        features_mv_shortcut = features_mv.clone()
        features_mv = self.mv_norm1(features_mv)
        mv_q = self.Wq(features_mv)
        mv_k = self.Wk(features_mv)
        mv_v = self.Wv(features_mv)

        mv_q = rearrange(mv_q, "(b v) (h w) c -> (b v) c h w", b=b, v=v, h=h, w=w, c=c).contiguous()
        mv_k = rearrange(mv_k, "(b v) (h w) c -> (b v) c h w", b=b, v=v, h=h, w=w, c=c).contiguous()
        mv_v = rearrange(mv_v, "(b v) (h w) c -> (b v) c h w", b=b, v=v, h=h, w=w, c=c).contiguous()

        intrinsics_curr = intrinsics.clone()
        intrinsics_curr[:, :, :2] = intrinsics_curr[:, :, :2] / self.upsample_factor

        mv_q_curr = list(torch.unbind(rearrange(mv_q, "(b v) ... -> b v ...", b=b, v=v), dim=1))
        mv_k_curr = list(torch.unbind(rearrange(mv_k, "(b v) ... -> b v ...", b=b, v=v), dim=1))
        mv_v_curr = list(torch.unbind(rearrange(mv_v, "(b v) ... -> b v ...", b=b, v=v), dim=1))

        intrinsics_curr = list(torch.unbind(intrinsics_curr, dim=1))
        extrinsics_curr = list(torch.unbind(extrinsics, dim=1))

        (ref_features, ref_intrinsics, ref_extrinsics, k_tgt_features, v_tgt_features, tgt_intrinsics, tgt_extrinsics,) = (
            batch_features_camera_parameters(mv_q_curr, mv_k_curr, mv_v_curr, intrinsics_curr, extrinsics_curr, nn_matrix=nn_matrix, ))

        b_new, v_1, _, h, w = k_tgt_features.size()

        if warp_indices is None:
            pose_curr = torch.matmul(tgt_extrinsics.inverse(), ref_extrinsics.unsqueeze(1))
            depth_candidates_curr = (depth_candidates.unsqueeze(1).expand(-1, k_tgt_features.size(1), -1, h, w).contiguous().view(-1, d, h, w))

            intrinsics_input = torch.stack(intrinsics_curr, dim=1).view(-1, 3, 3)
            intrinsics_input = intrinsics_input.unsqueeze(1).expand(-1, k_tgt_features.size(1), -1, -1)
            warp_indices, valid = warp_indices_with_pose_depth(
                rearrange(intrinsics_input, "b v ... -> (b v) ..."),
                rearrange(pose_curr, "b v ... -> (b v) ..."),
                1.0 / depth_candidates_curr,
                grid_sample_disable_cudnn=self.grid_sample_disable_cudnn, )
            warp_indices = rearrange(warp_indices, "(b v v_1) h w d-> (b v v_1) (h w) d", b=b, v=v, v_1=v_1, d=d, h=h, w=w)

        Q = ref_features.unsqueeze(1).expand(-1, v_1, -1, -1, -1) * self.scale
        Q = rearrange(Q, "(b v) v_1 c h w -> (b v v_1) (h w) c", b=b, v=v, v_1=v_1, c=c, h=h, w=w)
        K = rearrange(k_tgt_features, "(b v) v_1 c h w -> (b v v_1) (h w) c", b=b, v=v, v_1=v_1, c=c, h=h, w=w).transpose(-2, -1)
        V = rearrange(v_tgt_features, "(b v) v_1 c h w -> (b v v_1) (h w) c", b=b, v=v, v_1=v_1, c=c, h=h, w=w)

        smm_index = warp_indices.int()
        cost_volume_attn = SMM_QmK.apply(Q, K, smm_index)

        K_pe = self.k_pe.expand(b * v * v_1, h * w, -1)
        current_topk = warp_indices.shape[-1]
        Q_pe = Q.sum(dim=-1, keepdim=True).expand(-1, -1, current_topk)
        cost_volume_attn = cost_volume_attn + Q_pe * K_pe

        r_pe = self.rpe.expand(b * v * v_1, h * w, -1)
        cost_volume_attn = cost_volume_attn + r_pe

        cost_volume_attn = rearrange(cost_volume_attn, "(b v v_1) (h w) k -> (b v v_1) k h w", b=b, v=v, v_1=v_1, k=current_topk, h=h, w=w)
        set_num_views(self.Sim_UNet, num_views=v)
        
        cost_volume_attn = self.Sim_UNet(torch.concat((depth_in, cost_volume_attn), dim=1))
        cost_volume_attn = rearrange(cost_volume_attn, "(b v v_1) k h w -> (b v v_1) (h w) k", b=b, v=v, v_1=v_1, k=current_topk, h=h, w=w)

        cost_volume_attn = self.softmax(cost_volume_attn)

        # Depth Probability Boosting Strategy (Multiplicative Fusion)
        if depth_prob is not None:
            cost_volume_attn = (cost_volume_attn * depth_prob)
            cost_volume_attn = (cost_volume_attn + self.eps) / (cost_volume_attn.sum(dim=-1, keepdim=True) + self.eps)

        depth_prob = cost_volume_attn

        smm_index = warp_indices.int()
        out = SMM_AmV.apply(cost_volume_attn, V, smm_index)

        V_pe = self.v_pe.expand(b * v * v_1, h * w, -1)

        out = out + (cost_volume_attn * V_pe).sum(dim=-1, keepdim=True)

        out = rearrange(out, "(b v v_1) (h w) c -> (b v) v_1 (h w) c", b=b, v=v, v_1=v_1, c=c, h=h, w=w).mean(1)
        out = self.proj(out)
        cost_volume_out = out + features_mv_shortcut

        cost_volume_out = cost_volume_out + self.convffn(self.mv_norm2(cost_volume_out), x_size)

        cost_volume_out = rearrange(cost_volume_out, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=c)

        return cost_volume_out, warp_indices, depth_prob





# -------------------------------------------------------------------------
# The class naming is kept consistent with DepthSplat
# -------------------------------------------------------------------------
class MultiViewUniMatch(nn.Module):
    def __init__(
            self,
            num_scales=1,
            feature_channels=128,
            upsample_factor=4,
            lowest_feature_resolution=8,
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            num_depth_candidates=257,
            vit_type="vits",
            unet_channel_mult=[1, 1, 1, 1, 1],
            unet_num_res_blocks=1,
            unet_attn_resolutions=[16],
            window_size=16,
            mlp_ratio=2,
            convffn_kernel_size=5,
            gaussian_num_topk=[256, 256, 128, 128, 64, 64],
            grid_sample_disable_cudnn=False,
            **kwargs,
    ):
        super().__init__()


        self.feature_channels = feature_channels
        self.num_scales = num_scales

        ###### DL3DV->8, others->4
        self.lowest_feature_resolution = lowest_feature_resolution

        ###### DL3DV->8, others->4
        self.upsample_factor = upsample_factor

        self.vit_type = vit_type
        self.window_size = window_size

        channels = feature_channels
        gaussian_channels = 2 * channels
        dpbu_channels = 2 * channels
        dpbu_channels_u2 = channels // 2
        dpbu_channels_u4 = channels // 4

        # Keep the effective candidate count identical to the original code.
        self.num_depth_candidates = num_depth_candidates
        self.num_depth_candidates_u2 = num_depth_candidates // 2 + 1
        self.num_depth_candidates_u4 = num_depth_candidates // 4 + 1


        vit_feature_channel_dict = {"vits": 384, "vitb": 768, "vitl": 1024}
        mono_feature_channels = vit_feature_channel_dict[vit_type]
        cnn_feature_channels = 128
        mv_transformer_feature_channels = 128

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39],
        }
        self.model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        }
        self.encoder_type = vit_type
        mono_out_channels = self.model_configs[vit_type]['features']

        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=num_scales,
            downsample_factor=upsample_factor,
            lowest_scale=lowest_feature_resolution,
            return_all_scales=True,
        )
        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
        )

        self.mv_pyramid_u2 = ViTFeaturePyramid(in_channels=dpbu_channels, scale_factors=[2])
        self.cvo_pyramid_u2 = ViTFeaturePyramid(in_channels=dpbu_channels, scale_factors=[2])
        self.mv_pyramid_u4 = ViTFeaturePyramid(in_channels=dpbu_channels // 2, scale_factors=[2])
        self.cvo_pyramid_u4 = ViTFeaturePyramid(in_channels=dpbu_channels_u2, scale_factors=[2])
        self.features_cnn_u2 = ViTFeaturePyramid(in_channels=cnn_feature_channels - 32, scale_factors=[2])

        self.depth_anything = DepthAnything(self.model_configs[vit_type])

        mv_in_channels = cnn_feature_channels + mv_transformer_feature_channels + mono_feature_channels + mono_out_channels
        self.conv_mv = nn.Conv2d(mv_in_channels, dpbu_channels, 1, 1, 0)

        mv_u2_in_channels = (
            dpbu_channels // 2
            + dpbu_channels // 2
            + cnn_feature_channels - 32
            + mono_feature_channels
            + mono_out_channels
        )
        self.conv_mv_u2 = nn.Conv2d(mv_u2_in_channels, dpbu_channels_u2, 1, 1, 0)

        mv_u4_in_channels = (
            dpbu_channels_u2 // 2
            + dpbu_channels // 4
            + (cnn_feature_channels - 32) // 2
            + mono_feature_channels
            + mono_out_channels
        )
        self.conv_mv_u4 = nn.Conv2d(mv_u4_in_channels, dpbu_channels_u4, 1, 1, 0)

        # Depth Probability Boosting Units (DPBUs) (1st)
        self.wiea_f1_1 = WarpIndexEpipolarAttention(
            dpbu_channels, self.num_depth_candidates, self.upsample_factor, upsample_factor,
            mlp_ratio, convffn_kernel_size, grid_sample_disable_cudnn, num_heads=1, qkv_bias=True
        )
        self.wiea_f1_2 = WarpIndexEpipolarAttention(
            dpbu_channels, self.num_depth_candidates, self.upsample_factor, upsample_factor,
            mlp_ratio, convffn_kernel_size, grid_sample_disable_cudnn, num_heads=1, qkv_bias=True
        )

        # Depth Probability Boosting Units (DPBUs) (2-th)
        self.wiea_f2_1 = WarpIndexEpipolarAttention(
            dpbu_channels_u2, self.num_depth_candidates_u2, self.upsample_factor, upsample_factor // 2,
            mlp_ratio, convffn_kernel_size, grid_sample_disable_cudnn, num_heads=1, qkv_bias=True
        )
        self.wiea_f2_2 = WarpIndexEpipolarAttention(
            dpbu_channels_u2, self.num_depth_candidates_u2, self.upsample_factor, upsample_factor // 2,
            mlp_ratio, convffn_kernel_size, grid_sample_disable_cudnn, num_heads=1, qkv_bias=True
        )

        # Depth Probability Boosting Units (DPBUs) (3-th)
        self.wiea_f3_1 = WarpIndexEpipolarAttention(
            dpbu_channels_u4, self.num_depth_candidates_u4, self.upsample_factor, upsample_factor // 4,
            mlp_ratio, convffn_kernel_size, grid_sample_disable_cudnn, num_heads=1, qkv_bias=True
        )
        self.wiea_f3_2 = WarpIndexEpipolarAttention(
            dpbu_channels_u4, self.num_depth_candidates_u4, self.upsample_factor, upsample_factor // 4,
            mlp_ratio, convffn_kernel_size, grid_sample_disable_cudnn, num_heads=1, qkv_bias=True
        )

        # Residual depth refinement head.
        res_depth_channels = dpbu_channels_u4 + self.num_depth_candidates_u4 + 1
        res_depth_head_channels = dpbu_channels_u4
        self.res_depth_head = nn.Sequential(
            nn.Conv2d(res_depth_channels, res_depth_head_channels, 3, 1, 1),
            nn.GroupNorm(8, res_depth_head_channels),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=res_depth_head_channels,
                model_channels=res_depth_head_channels,
                out_channels=res_depth_head_channels,
                num_res_blocks=unet_num_res_blocks,
                attention_resolutions=unet_attn_resolutions,
                channel_mult=unet_channel_mult,
                num_head_channels=res_depth_head_channels // 4,
                dims=2,
                postnorm=False,
                num_frames=2,
                use_cross_view_self_attn=True,
            ),
            nn.Conv2d(res_depth_head_channels, self.num_depth_candidates_u4, 3, 1, 1),
        )

        relative_position_index_sa = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_sa)

        self.conv_image = nn.Conv2d(3, channels // 16, 1, 1, 0, padding_mode="replicate")

        gaussian_in_channels = channels + cnn_feature_channels + dpbu_channels + 16 * dpbu_channels_u4 + 16
        self.gaussian_fuse = nn.Conv2d(gaussian_in_channels, gaussian_channels, 1, 1, 0, padding_mode="replicate")

        self.gaussian_layers = nn.ModuleList([
            GaussianFocusedLayer(
                gaussian_channels,
                window_size=self.window_size,
                num_topk=gaussian_num_topk,
                shift_size=0 if layer_idx % 2 == 0 else self.window_size // 2,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=mlp_ratio,
                layer_id=layer_idx,
                num_heads=4,
                qkv_bias=True,
            )
            for layer_idx in range(6)
        ])

        self.gaussian_conv = nn.Conv2d(gaussian_channels, 16 * channels, 1, 1, 0, padding_mode="replicate")
        self.pixelshuffle_2 = nn.PixelShuffle(2)
        self.pixelshuffle_4 = nn.PixelShuffle(4)
        self.pixelunshuffle_2 = nn.PixelUnshuffle(2)
        self.pixelunshuffle_4 = nn.PixelUnshuffle(4)

    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        return relative_coords.sum(-1)

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -(self.window_size // 2)),
            slice(-(self.window_size // 2), None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -(self.window_size // 2)),
            slice(-(self.window_size // 2), None),
        )

        cnt = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def normalize_images(self, images):
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)
        return (images - mean) / std

    def extract_feature(self, images):
        concat = rearrange(images, "b v c h w -> (b v) c h w")
        return self.backbone(concat)[::-1]

    def extract_mono_feature(self, images):
        b, v, _, ori_h, ori_w = images.shape
        resize_h, resize_w = ori_h // 14 * 14, ori_w // 14 * 14

        images = rearrange(images, "b v c h w -> (b v) c h w")
        images = F.interpolate(images, (resize_h, resize_w), mode="bilinear", align_corners=True)

        features = self.depth_anything.pretrained.get_intermediate_layers(
            images,
            self.intermediate_layer_idx[self.vit_type],
            return_class_token=True,
        )
        mono_intermediate_features = list(list(zip(*features))[0])

        patch_size = self.depth_anything.pretrained.patch_size
        patch_h, patch_w = resize_h // patch_size, resize_w // patch_size
        _, mono_path1, _, _, _, mono_disp = self.depth_anything.depth_head.forward(
            features, patch_h, patch_w, return_intermediate=True, patch_size=patch_size
        )

        features_mono = mono_intermediate_features[-1].reshape(
            images.shape[0], resize_h // 14, resize_w // 14, -1
        ).permute(0, 3, 1, 2).contiguous()

        features_mono = F.interpolate(
            features_mono,
            (ori_h // self.lowest_feature_resolution, ori_w // self.lowest_feature_resolution),
            mode="bilinear",
            align_corners=True,
        )
        features_mono_u2 = F.interpolate(
            features_mono,
            (ori_h // (self.lowest_feature_resolution // 2), ori_w // (self.lowest_feature_resolution // 2)),
            mode="bilinear",
            align_corners=True,
        )
        features_mono_u4 = F.interpolate(
            features_mono,
            (ori_h // (self.lowest_feature_resolution // 4), ori_w // (self.lowest_feature_resolution // 4)),
            mode="bilinear",
            align_corners=True,
        )

        mono_out = F.interpolate(
            mono_path1,
            (ori_h // self.lowest_feature_resolution, ori_w // self.lowest_feature_resolution),
            mode="bilinear",
            align_corners=True,
        )
        mono_out_u2 = F.interpolate(
            mono_path1,
            (ori_h // (self.lowest_feature_resolution // 2), ori_w // (self.lowest_feature_resolution // 2)),
            mode="bilinear",
            align_corners=True,
        )
        mono_out_u4 = F.interpolate(
            mono_path1,
            (ori_h // (self.lowest_feature_resolution // 4), ori_w // (self.lowest_feature_resolution // 4)),
            mode="bilinear",
            align_corners=True,
        )

        mono_disp = F.interpolate(
            mono_disp,
            (ori_h // self.lowest_feature_resolution, ori_w // self.lowest_feature_resolution),
            mode="bilinear",
            align_corners=True,
        )

        features_mono_list = [features_mono, features_mono_u2, features_mono_u4]
        mono_out_list = [mono_out, mono_out_u2, mono_out_u4]
        return features_mono_list, mono_out_list, mono_disp

    def num_neighbor_views(self, num_views, nn_matrix):
        return nn_matrix.size(-1) - 1 if nn_matrix is not None else num_views - 1

    def scale_intrinsics_to_pixels(self, intrinsics, ori_h, ori_w):
        intrinsics = intrinsics.clone()
        intrinsics[:, :, 0] *= ori_w
        intrinsics[:, :, 1] *= ori_h
        return intrinsics

    def feature_extraction(self, images, attn_splits, b, v, nn_matrix):
        features_list_cnn = self.extract_feature(images)
        features_cnn = features_list_cnn[0]

        features_mono_list, mono_out_list, mono_disp = self.extract_mono_feature(images)

        features_cnn_pos = mv_feature_add_position(features_cnn, attn_splits, self.feature_channels)
        features_list = list(torch.unbind(
            rearrange(features_cnn_pos, "(b v) c h w -> b v c h w", b=b, v=v),
            dim=1,
        ))
        features_list_mv = self.transformer(features_list, attn_num_splits=attn_splits, nn_matrix=nn_matrix)
        features_mv = torch.stack(features_list_mv, dim=1)
        features_mv = rearrange(features_mv, "b v c h w -> (b v) c h w")

        return features_list_cnn, features_cnn, features_mono_list, mono_out_list, mono_disp, features_mv

    def build_initial_depth_candidates(self, min_depth, max_depth, num_candidates, h, w, dtype_ref):
        linear_space = torch.linspace(0, 1, num_candidates, device=dtype_ref.device, dtype=dtype_ref.dtype).view(1, num_candidates, 1, 1)
        depth_candidates = min_depth.view(-1, 1, 1, 1) + linear_space * (max_depth.view(-1, 1, 1, 1) - min_depth.view(-1, 1, 1, 1))
        return depth_candidates.expand(-1, -1, h, w)

    def build_refined_depth_candidates(self, center_depth, min_depth, max_depth, full_candidates, stage_candidates, shrink_divisor, dtype_ref):
        depth_interval = ((max_depth - min_depth) / (full_candidates - 1) / shrink_divisor).view(-1, 1, 1, 1)
        half_span = (stage_candidates - 1) / 2

        depth_range_min = (center_depth - depth_interval * half_span).clamp(min=min_depth.view(-1, 1, 1, 1))
        depth_range_max = (center_depth + depth_interval * half_span).clamp(max=max_depth.view(-1, 1, 1, 1))
        depth_range_mid = (depth_range_min + depth_range_max) / 2
        depth_interval_true = (depth_range_max - depth_range_min) / (stage_candidates - 1)

        linear_space = torch.linspace(
            -half_span,
            half_span,
            stage_candidates,
            device=dtype_ref.device,
            dtype=dtype_ref.dtype,
        ).view(1, stage_candidates, 1, 1)
        return depth_range_mid + linear_space * depth_interval_true

    def expand_depth_for_neighbors(self, depth, b, v, v_1, nn_matrix):
        # depth: (b*v, 1, h, w) or already (b*v*v_1, 1, h, w)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)

        if depth.shape[0] == b * v * v_1:
            return depth.contiguous()

        if depth.shape[0] == b * v:
            return rearrange(
                depth.unsqueeze(1).expand(-1, v_1, -1, -1, -1),
                "(b v) v_1 c h w -> (b v v_1) c h w",
                b=b,
                v=v,
                v_1=v_1,
            ).contiguous()

        raise RuntimeError(
            f"Unexpected depth shape {tuple(depth.shape)}, "
            f"expected {b*v} or {b*v*v_1}"
        )

    def depth_prob_to_depth(self, depth_prob, depth_candidates, b, v, v_1, d, h, w):
        match_prob = rearrange(
            depth_prob,
            "(b v v_1) (h w) d -> (b v) v_1 d h w",
            b=b,
            v=v,
            v_1=v_1,
            d=d,
            h=h,
            w=w,
        ).mean(1)
        depth = (match_prob * depth_candidates).sum(dim=1, keepdim=True)
        return match_prob, depth

    def upsample_depth(self, depth, scale_factor):
        return F.interpolate(depth, scale_factor=scale_factor, mode="bilinear", align_corners=True)

    def run_wiea_stage_pair(self, stage_input, stage_seed_depth, depth_candidates, first_block, second_block, extrinsics, intrinsics, nn_matrix, b, v, v_1, d, h, w):
        depth_prob = None
        warp_indices = None

        stage_output, warp_indices, depth_prob = first_block(
            stage_input,
            stage_seed_depth,
            warp_indices,
            depth_prob,
            depth_candidates,
            extrinsics,
            intrinsics,
            nn_matrix,
        )

        first_match_prob, first_depth = self.depth_prob_to_depth(depth_prob, depth_candidates, b, v, v_1, d, h, w)

        second_seed_depth = self.expand_depth_for_neighbors(first_depth, b, v, v_1, nn_matrix)
        stage_output, warp_indices, depth_prob = second_block(
            stage_output,
            second_seed_depth,
            warp_indices,
            depth_prob,
            depth_candidates,
            extrinsics,
            intrinsics,
            nn_matrix,
        )

        # depth_prob, warp_indices = torch.topk(depth_prob, d // 2, dim=-1, largest=True, sorted=False)
        # depth_prob = torch.zeros((b * v * v_1, h * w, d), device=depth_prob.device).scatter(-1, warp_indices, depth_prob)
        # depth_prob = depth_prob / (depth_prob.sum(dim=-1, keepdim=True) + 1e-8)

        final_match_prob, final_depth = self.depth_prob_to_depth(depth_prob, depth_candidates, b, v, v_1, d, h, w)

        return {
            "stage_output": stage_output,
            "warp_indices": warp_indices,
            "depth_prob": depth_prob,
            "first_match_prob": first_match_prob,
            "first_depth": first_depth,
            "final_match_prob": final_match_prob,
            "final_depth": final_depth,
        }

    def pad_to_window_multiple(self, x):
        h_ori, w_ori = x.size()[-2:]
        mod = self.window_size

        if (h_ori % mod == 0) and (w_ori % mod == 0):
            return x, h_ori, w_ori, h_ori, w_ori

        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad

        x = torch.cat([x, torch.flip(x, [2])], dim=2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], dim=3)[:, :, :, :w]
        return x, h_ori, w_ori, h, w

    def run_gaussian_stack(self, gaussian_pred, h, w):
        w_pfa_values = [None, None]
        w_pfa_indices = [None, None]

        attn_mask = self.calculate_mask([h, w]).to(gaussian_pred.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        for layer in self.gaussian_layers:
            gaussian_pred, w_pfa_values, w_pfa_indices = layer(
                gaussian_pred,
                w_pfa_values,
                w_pfa_indices,
                params,
            )
        return gaussian_pred

    def forward(self, images, attn_splits_list=None, intrinsics=None, min_depth=1.0 / 100, max_depth=1.0 / 0.5, extrinsics=None, nn_matrix=None, **kwargs):
        results_dict = {}
        depth_preds = []
        gaussian_preds = []

        images = self.normalize_images(images)
        b, v, _, ori_h, ori_w = images.shape
        v_1 = self.num_neighbor_views(v, nn_matrix)

        intrinsics = self.scale_intrinsics_to_pixels(intrinsics, ori_h, ori_w)
        min_depth = min_depth.view(-1)
        max_depth = max_depth.view(-1)

        attn_splits = attn_splits_list[0]
        features_list_cnn, features_cnn, features_mono_list, mono_out_list, mono_disp, features_mv = self.feature_extraction(images, attn_splits, b, v, nn_matrix)

        # ------------------------------------------------------------------
        # Stage 1 (coarsest depth estimation)
        # ------------------------------------------------------------------
        features_mv_stage1_flat = torch.cat((features_cnn, features_mv, features_mono_list[0], mono_out_list[0]), dim=1)
        features_mv_stage1_flat = self.conv_mv(features_mv_stage1_flat)

        _, c_stage1, h_d4, w_d4 = features_mv_stage1_flat.shape
        depth_candidates_f1 = self.build_initial_depth_candidates(min_depth, max_depth, self.num_depth_candidates, h_d4, w_d4, features_mv)

        features_mv_stage1 = rearrange(features_mv_stage1_flat, "(b v) c h w -> b v c h w", b=b, v=v, h=h_d4, w=w_d4, c=c_stage1)
        stage1_seed_depth = rearrange(mono_disp.expand(-1, v_1, -1, -1),  "(b v) v_1 h w -> (b v v_1) 1 h w", b=b, v=v, v_1=v_1, h=h_d4, w=w_d4)

        stage1 = self.run_wiea_stage_pair(features_mv_stage1, stage1_seed_depth, depth_candidates_f1, self.wiea_f1_1, self.wiea_f1_2, extrinsics, intrinsics, nn_matrix, b, v, v_1, self.num_depth_candidates, h_d4, w_d4)
        depth_mvs_f1 = stage1["final_depth"]
        depth_mvs_f1_u2 = self.upsample_depth(depth_mvs_f1, scale_factor=2)
        depth_f1 = self.upsample_depth(depth_mvs_f1, scale_factor=self.upsample_factor)

        # ------------------------------------------------------------------
        # Stage 2
        # ------------------------------------------------------------------
        cost_volume_out_stage1_flat = rearrange(stage1["stage_output"], "b v c h w -> (b v) c h w", b=b, v=v, h=h_d4, w=w_d4, c=c_stage1)
        features_mv_stage1_flat = rearrange(features_mv_stage1, "b v c h w -> (b v) c h w", b=b, v=v, h=h_d4, w=w_d4, c=c_stage1)

        cost_volume_out_u2 = self.cvo_pyramid_u2(cost_volume_out_stage1_flat)[0]
        features_mv_in_u2 = self.mv_pyramid_u2(features_mv_stage1_flat)[0]
        cost_volume_out_u2 = torch.cat((cost_volume_out_u2, features_mv_in_u2, features_list_cnn[1], features_mono_list[1], mono_out_list[1]), dim=1)
        cost_volume_out_u2 = self.conv_mv_u2(cost_volume_out_u2)

        _, c_u2, h_u2, w_u2 = cost_volume_out_u2.shape
        depth_candidates_f2 = self.build_refined_depth_candidates(depth_mvs_f1_u2, min_depth, max_depth, self.num_depth_candidates, self.num_depth_candidates_u2, shrink_divisor=2, dtype_ref=features_mv)

        cost_volume_out_u2 = rearrange(cost_volume_out_u2, "(b v) c h w -> b v c h w", b=b, v=v, h=h_u2, w=w_u2, c=c_u2)
        stage2_seed_depth = self.expand_depth_for_neighbors(depth_mvs_f1_u2, b, v, v_1, nn_matrix)

        stage2 = self.run_wiea_stage_pair(cost_volume_out_u2, stage2_seed_depth, depth_candidates_f2, self.wiea_f2_1, self.wiea_f2_2, extrinsics, intrinsics, nn_matrix, b, v, v_1, self.num_depth_candidates_u2, h_u2, w_u2)
        depth_mvs_f2 = stage2["final_depth"]
        depth_mvs_f2_u2 = self.upsample_depth(depth_mvs_f2, scale_factor=2)
        depth_f2 = self.upsample_depth(depth_mvs_f2, scale_factor=self.upsample_factor // 2)

        # ------------------------------------------------------------------
        # Stage 3
        # ------------------------------------------------------------------
        cost_volume_out_stage2_flat = rearrange(stage2["stage_output"], "b v c h w -> (b v) c h w", b=b, v=v, h=h_u2, w=w_u2, c=c_u2)
        cost_volume_out_u4 = self.cvo_pyramid_u4(cost_volume_out_stage2_flat)[0]
        features_mv_in_u4 = self.mv_pyramid_u4(features_mv_in_u2)[0]
        features_list_cnn_u2 = self.features_cnn_u2(features_list_cnn[1])[0]

        cost_volume_out_u4 = torch.cat((cost_volume_out_u4, features_mv_in_u4, features_list_cnn_u2, features_mono_list[2], mono_out_list[2]), dim=1)
        cost_volume_out_u4 = self.conv_mv_u4(cost_volume_out_u4)

        _, c_u4, h_u4, w_u4 = cost_volume_out_u4.shape
        depth_candidates_f3 = self.build_refined_depth_candidates(depth_mvs_f2_u2, min_depth, max_depth, self.num_depth_candidates, self.num_depth_candidates_u4, shrink_divisor=4, dtype_ref=features_mv)

        cost_volume_out_u4 = rearrange(cost_volume_out_u4, "(b v) c h w -> b v c h w", b=b, v=v, h=h_u4, w=w_u4, c=c_u4)
        stage3_seed_depth = self.expand_depth_for_neighbors(depth_mvs_f2_u2, b, v, v_1, nn_matrix)

        stage3 = self.run_wiea_stage_pair(cost_volume_out_u4, stage3_seed_depth, depth_candidates_f3, self.wiea_f3_1, self.wiea_f3_2, extrinsics, intrinsics, nn_matrix, b, v, v_1, self.num_depth_candidates_u4, h_u4, w_u4)
        depth_mvs_f3 = stage3["final_depth"]
        match_prob_f3 = stage3["final_match_prob"]
        depth_f3 = self.upsample_depth(depth_mvs_f3, scale_factor=self.upsample_factor // 4)

        # ------------------------------------------------------------------
        # Residual depth refinement and final depth fusion
        # ------------------------------------------------------------------
        cost_volume_out = rearrange(stage3["stage_output"], "b v c h w -> (b v) c h w", b=b, v=v, h=h_u4, w=w_u4, c=c_u4)
        res_depth_feature = torch.cat((cost_volume_out, match_prob_f3, depth_mvs_f3), dim=1)
        
        set_num_views(self.res_depth_head, num_views=v)
        res_match_prob = F.softmax(self.res_depth_head(res_depth_feature), dim=1)
        res_depth_mvs = (res_match_prob * depth_candidates_f3).sum(dim=1, keepdim=True)
        res_depth = self.upsample_depth(res_depth_mvs, scale_factor=self.upsample_factor // 4)

        ### Linearly combined to obtain better depth estimation results
        depth = 0.1 * depth_f1 + 0.1 * depth_f2 + 0.4 * depth_f3 + 0.4 * res_depth
        depth = torch.nan_to_num(depth, nan=0.01, posinf=2., neginf=0.01)

        depth = depth.clamp(min=min_depth.view(-1, 1, 1, 1), max=max_depth.view(-1, 1, 1, 1),)
        depth_preds.append(depth)

        for i in range(len(depth_preds)):
            depth_pred = 1.0 / depth_preds[i].squeeze(1)
            depth_preds[i] = rearrange(depth_pred, "(b v) ... -> b v ...", b=b, v=v)

        results_dict.update({"depth_preds": depth_preds})

        # ------------------------------------------------------------------
        # Gaussian prediction head
        # ------------------------------------------------------------------
        images_flat = rearrange(images, "b v ... -> (b v) ...", b=b, v=v)
        images_feature = self.conv_image(images_flat)
        images_feature = self.pixelunshuffle_4(images_feature)

        ###### DL3DV->0.5, others->1
        images_feature = F.interpolate(images_feature, scale_factor=4 / self.upsample_factor, mode="bilinear", align_corners=True,)

        cost_volume_out = self.pixelunshuffle_4(cost_volume_out)
        depth_d4 = self.pixelunshuffle_4(depth)

        ######DL3DV->0.5, others->1
        depth_d4 = F.interpolate(depth_d4, scale_factor=4 / self.upsample_factor, mode="bilinear", align_corners=True,)

        features_gaussian_concat = torch.cat((images_feature, features_cnn, features_mv_stage1_flat, cost_volume_out, depth_d4), dim=1)
        gaussian_pred = self.gaussian_fuse(features_gaussian_concat)

        gaussian_pred, h_ori, w_ori, h_pad, w_pad = self.pad_to_window_multiple(gaussian_pred)
        gaussian_pred = self.run_gaussian_stack(gaussian_pred, h_pad, w_pad)
        gaussian_pred = gaussian_pred[..., :h_ori, :w_ori]

        gaussian_pred = self.gaussian_conv(gaussian_pred)
        gaussian_pred = self.pixelshuffle_4(gaussian_pred)

        #####DL3DV->2, others->1
        gaussian_pred = F.interpolate(gaussian_pred, scale_factor=self.upsample_factor // 4, mode="bilinear", align_corners=True,)

        gaussian_preds.append(gaussian_pred)
        results_dict.update({"gaussian_preds": gaussian_preds})
        return results_dict


def set_num_views(module, num_views):
    if isinstance(module, AttentionBlock):
        module.attention.n_frames = num_views
    elif (
            isinstance(module, nn.ModuleList)
            or isinstance(module, nn.Sequential)
            or isinstance(module, nn.Module)
    ):
        for submodule in module.children():
            set_num_views(submodule, num_views)