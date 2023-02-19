# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pprint import pformat
from typing import List

import sys
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN
from timm.models.layers import trunc_normal_

import encoder
from decoder import LightDecoder


class SparK(nn.Module):
    def __init__(
            self, sparse_encoder: encoder.SparseEncoder, dense_decoder: LightDecoder,
            mask_ratio=0.6, densify_norm='bn', sbn=False, hierarchy=4,
    ):
        super().__init__()
        input_size, downsample_raito = sparse_encoder.input_size, sparse_encoder.downsample_raito
        self.downsample_raito = downsample_raito
        self.fmap_size = input_size // downsample_raito
        self.mask_ratio = mask_ratio
        self.len_keep = round(self.fmap_size * self.fmap_size * (1 - mask_ratio))
        
        self.sparse_encoder = sparse_encoder
        self.dense_decoder = dense_decoder
        
        self.sbn = sbn
        self.hierarchy = hierarchy
        self.densify_norm_str = densify_norm.lower()
        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()
        
        # build the `densify` layers
        e_width, d_width = self.sparse_encoder.fea_dim, self.dense_decoder.width
        for i in range(self.hierarchy):
            if self.densify_norm_str == 'bn':
                densify_norm = (encoder.SparseSyncBatchNorm2d if self.sbn else encoder.SparseBatchNorm2d)(e_width)
            elif self.densify_norm_str == 'ln':
                densify_norm = encoder.SparseConvNeXtLayerNorm(e_width, data_format='channels_first', sparse=True)
            else:
                densify_norm = nn.Identity()
            self.densify_norms.append(densify_norm)
            
            if i == 0 and e_width == d_width:
                densify_proj = nn.Identity()    # todo: NOTE THAT CONVNEXT-S WOULD USE THIS, because it has a width of 768 that equals to the decoder's width 768
                print(f'[mid, py={self.hierarchy}][densify {i} proj]: use nn.Identity()')
            else:
                kernel_size = 1 if i <= 0 else 3
                densify_proj = nn.Conv2d(e_width, d_width, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True)
                print(f'[mid, py={self.hierarchy}][densify {i} proj]: k={kernel_size}, #para = {sum(x.numel() for x in densify_proj.parameters()) / 1e6:.2f}')
            self.densify_projs.append(densify_proj)
            
            p = nn.Parameter(torch.zeros(1, e_width, 1, 1))
            trunc_normal_(p, mean=0, std=.02, a=-.02, b=.02)
            self.mask_tokens.append(p)
            e_width //= 2
            d_width //= 2
        
        print(f'[mid, py={self.hierarchy}][mask_tokens]: {tuple(p.numel() for p in self.mask_tokens)}')
        
        m = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        s = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
        self.register_buffer('imn_m', m)
        self.register_buffer('imn_s', s)
        self.register_buffer('norm_black', torch.zeros(1, 3, input_size, input_size))
        self.vis_active = self.vis_active_ex = self.vis_inp = self.vis_inp_mask = ...
    
    def mask(self, B: int, device, generator=None):
        f: int = self.fmap_size
        idx = torch.rand(B, f * f, generator=generator).argsort(dim=1)
        idx = idx[:, :self.len_keep].to(device)  # (B, len_keep)
        return torch.zeros(B, f * f, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, f, f)
    
    def forward(self, inp_bchw: torch.Tensor, active_b1ff=None):
        # step1. Mask
        if active_b1ff is None:
            active_b1ff: torch.BoolTensor = self.mask(inp_bchw.shape[0], inp_bchw.device)  # (B, 1, f, f)
        encoder._cur_active = active_b1ff    # (B, 1, f, f)
        active_b1hw = active_b1ff.repeat_interleave(self.downsample_raito, 2).repeat_interleave(self.downsample_raito, 3)  # (B, 1, H, W)
        masked_bchw = inp_bchw * active_b1hw
        
        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        fea_bcffs: List[torch.Tensor] = self.sparse_encoder(masked_bchw, hierarchy=self.hierarchy)
        fea_bcffs.reverse()  # after reversion: from the smallest feature map to the largest
        
        # step3. Densify: get hierarchical dense features for decoding
        cur_active = active_b1ff     # (B, 1, f, f)
        to_dec = []
        for i, bcff in enumerate(fea_bcffs):  # from the smallest feature map to the largest
            if bcff is not None:
                bcff = self.densify_norms[i](bcff)
                mask_tokens = self.mask_tokens[i].expand_as(bcff)
                bcff = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)   # fill in empty (non-active) positions with [mask] tokens
                bcff: torch.Tensor = self.densify_projs[i](bcff)
            to_dec.append(bcff)
            cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # dilate the mask map, from (B, 1, f, f) to (B, 1, H, W)
        
        # step4. Decode and reconstruct
        rec_bchw = self.dense_decoder(to_dec)
        recon_loss = self.reconstruction_loss(inp_bchw, rec_bchw, active_b1ff)
        
        return active_b1hw, rec_bchw, recon_loss
    
    def reconstruction_loss(self, inp, rec, active):  # active: (B, 1, f, f)
        inp, rec = self.patchify(inp), self.patchify(rec)   # inp and rec: (B, L = f*f, N = C*downsample_raito**2)
        mean = inp.mean(dim=-1, keepdim=True)
        var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** .5
        inp = (inp - mean) / var
        loss_spa = (rec - inp) ** 2
        
        loss_spa = loss_spa.mean(dim=2, keepdim=False)  # (B, L, C) => (B, L)
        non_active = active.logical_not().int().view(active.shape[0], -1)  # (B, 1, f, f) => (B, L)
        return loss_spa.mul_(non_active).sum() / (non_active.sum() + 1e-8)  # only on removed patches
    
    def patchify(self, bchw):
        p = self.downsample_raito
        h = w = self.fmap_size
        B, C = bchw.shape[:2]
        bchw = bchw.reshape(shape=(B, C, h, p, w, p))
        bchw = torch.einsum('bchpwq->bhwpqc', bchw)
        bln = bchw.reshape(shape=(B, h * w, C * p ** 2))  # (B, f*f, 3*downsample_raito**2)
        return bln
    
    def unpatchify(self, bln):
        p = self.downsample_raito
        h = w = self.fmap_size
        B, C = bln.shape[0], bln.shape[-1] // p ** 2
        bln = bln.reshape(shape=(B, h, w, p, p, C))
        bln = torch.einsum('bhwpqc->bchpwq', bln)
        bchw = bln.reshape(shape=(B, C, h * p, w * p))
        return bchw
    
    def __repr__(self):
        return (
            f'\n'
            f'[SparK.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[SparK.structure]: {super(SparK, self).__repr__().replace(SparK.__name__, "")}'
        )
    
    def get_config(self):
        return {
            # self
            'mask_ratio': self.mask_ratio,
            'en_de_norm': self.densify_norm_str,
            'sbn': self.sbn, 'hierarchy': self.hierarchy,
            
            # enc
            'input_size': self.sparse_encoder.input_size,
            # dec
            'dec_fea_dim': self.dense_decoder.width,
        }
    
    def state_dict(self, destination=None, prefix='', keep_vars=False, with_config=False):
        state = super(SparK, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if with_config:
            state['config'] = self.get_config()
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        config: dict = state_dict.pop('config', None)
        incompatible_keys = super(SparK, self).load_state_dict(state_dict, strict=strict)
        if config is not None:
            for k, v in self.get_config().items():
                ckpt_v = config.get(k, None)
                if ckpt_v != v:
                    err = f'[SparseMIM.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={ckpt_v})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err, file=sys.stderr)
        return incompatible_keys
    
    def denorm_for_vis(self, normalized_im):
        normalized_im = (normalized_im * self.imn_s).add_(self.imn_m)
        return torch.clamp(normalized_im, 0, 1)
