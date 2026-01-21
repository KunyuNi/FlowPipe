import gc
import math
import os
from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union
import torch
import numpy as np
import pandas as pd
from loguru import logger

import comp
from config import AgentConfig, Config
from ctxpipe.agent.dqn import Agent
from ctxpipe.attention import SelfAttentionBlock,FuseAttentionBlock
from sklearn.cluster import KMeans

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
def swiglu(x):
    """SwiGLU 激活：将通道对半切分，一半做值，一半做门控（SiLU）。"""
    x_val, x_gate = x.chunk(2, dim=-1)
    return x_val * F.silu(x_gate)


class LayerNorm(nn.Module):
    """简化的 LayerNorm 封装"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


# ---------------------------
# Attention Blocks
# ---------------------------
class SelfAttentionBlock(nn.Module):
    """自注意力块 - 处理单个 embedding"""
    def __init__(self, dim, dim_head=32, heads=8, ff_mult=4):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.norm = LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        ff_inner_dim = ff_mult * dim
        self.ff_in = nn.Linear(dim, ff_inner_dim * 2, bias=False)
        self.ff_out = nn.Linear(ff_inner_dim, dim, bias=False)

    def forward(self, x):
        x_res = x
        x = self.norm(x)

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads) * self.scale

        k, v = self.to_kv(x).chunk(2, dim=-1)

        sim = torch.einsum('b h i d, b j d -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # 关键改动：与旧版一致，对规范化后的 x 做 FFN
        ff = self.ff_out(swiglu(self.ff_in(x)))
        return out + ff


class FuseAttentionBlock(nn.Module):
    """融合注意力块 - 计算多种 embedding 的融合权重
    输入: x: [fn, B*N, D]
    输出: coef: [B*N, fn]
    """
    def __init__(self, dim, dim_fused):
        super().__init__()
        self.W = nn.Linear(dim, dim_fused, bias=False)
        self.f = nn.Linear(dim_fused * 2, 1)
        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        # x: [fn, B*N, D]
        f = self.W(x)   # [fn, B*N, dim_fused]
        fn = f.shape[0]

        coefs = []
        for i in range(fn):
            others = [j for j in range(fn) if j != i]
            scores = []
            for j in others:
                # 两两拼接打分 -> [B*N, 1]
                score = self.act(self.f(torch.cat([f[i], f[j]], dim=-1)))
                scores.append(score)
            ai = torch.mean(torch.stack(scores), dim=0)  # [B*N, 1]
            coefs.append(ai)

        coef = torch.cat(coefs, dim=-1)   # [B*N, fn]
        coef = F.softmax(coef, dim=-1)    # 融合权重
        return coef


class CrossAttentionBlock(nn.Module):
    """跨模态注意力块 - LLM embedding 与 POI embedding 交互（当前未启用）"""
    def __init__(self, dim, dim_fused, dim_head=32, heads=8, ff_mult=4):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.norm = LayerNorm(dim)
        self.norm_y = LayerNorm(dim_fused)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_fused, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        ff_inner_dim = ff_mult * dim
        # 改动①：FFN 与原先“2x隐藏维 + 门控(SwiGLU)”思路一致
        self.ff_in = nn.Linear(dim, ff_inner_dim * 2, bias=False)
        self.ff_out = nn.Linear(ff_inner_dim, dim, bias=False)

    def forward(self, x, y):
        x_res = x
        x = self.norm(x)
        y = self.norm_y(y)

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads) * self.scale

        k, v = self.to_kv(y).chunk(2, dim=-1)

        sim = torch.einsum('b h i d, b j d -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        ff = self.ff_out(swiglu(self.ff_in(x)))
        return out + ff
