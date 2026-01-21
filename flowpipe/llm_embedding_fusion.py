# fusion_script.py
# 这是一个完整的、可独立运行的脚本，用于融合四种LLM embedding和POI embedding（当前未启用POI交互）。
# 依赖: torch, einops, loguru (pip install torch einops loguru)
# 假设embedding文件存在于指定路径。如果没有，请先运行原项目的生成脚本。

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger


# ---------------------------
# Utils
# ---------------------------
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


# ---------------------------
# Fusion Model
# ---------------------------
class LLM_Fused(nn.Module):
    def __init__(self,dim=4096,cross_layer_num=3):
        super().__init__()
        self.dim = dim
        self.cross_layer_num = cross_layer_num
        self.attention_block1 = SelfAttentionBlock(dim=dim)
        self.attention_block2 = SelfAttentionBlock(dim=dim)
        self.attention_block3 = SelfAttentionBlock(dim=dim)
        self.attention_block4 = SelfAttentionBlock(dim=dim)

        self.fuse_attention = FuseAttentionBlock(dim=dim, dim_fused=2 * dim)

        self.cross_attention_fusion = nn.ModuleList([
            CrossAttentionBlock(dim=dim, dim_fused=dim) for _ in range(cross_layer_num)
        ])

    def forward(self,llm_e1,llm_e2,llm_e3,llm_e4):
        print(llm_e1.shape)
        # 形状保护（可选）：确保四路一致
        assert llm_e1.shape == llm_e2.shape == llm_e3.shape == llm_e4.shape, \
            f"Inconsistent shapes: {[t.shape for t in (llm_e1, llm_e2, llm_e3, llm_e4)]}"
        def norm_shape(t):
            t = t.to(torch.float32)
            if t.dim() == 2:
                t = t.unsqueeze(1)
            elif t.dim() == 1:
                t = t.view(1, 1, -1)
            return t

        llm_e1, llm_e2, llm_e3, llm_e4 = map(norm_shape, (llm_e1, llm_e2, llm_e3, llm_e4))
        # 四路自注意力
        x1 = self.attention_block1(llm_e1)
        x2 = self.attention_block2(llm_e2)
        x3 = self.attention_block3(llm_e3)
        x4 = self.attention_block4(llm_e4)

        # [4, B, N, D]
        out = torch.stack([x1, x2, x3, x4])

        # 展平到 [4, B*N, D]
        out1 = rearrange(out, 'fn b n d -> fn (b n) d')

        # 计算融合权重 [B*N, 4]
        coef = self.fuse_attention(out1)

        # 改动②：在展平空间做加权 -> [B*N, D]
        temp_flat = torch.zeros_like(out1[0])  # [B*N, D]
        for i in range(out1.shape[0]):         # 4 路
            temp_flat = temp_flat + out1[i] * coef[:, i].unsqueeze(-1)

        # 还原回 [B, N, D]
        B, N = x1.shape[0], x1.shape[1]
        z = rearrange(temp_flat, '(b n) d -> b n d', b=B, n=N)

        # 如需启用 POI 交互，取消以下注释并提供 poi_e
        # for cross_attention_fusion in self.cross_attention_fusion:
        #     z = cross_attention_fusion(z, poi_e)

        return z.squeeze()


# ---------------------------
# Main
# ---------------------------
def main():

    model = LLM_Fused(dim=4096, cross_layer_num=3)
    # styles = ['pipeline_component', 'raw_data', 'statistical_sampling', 'contextual_semantic']
    styles = ['contextual_semantic']

    embeddings = []

    for style in styles:
        embedding_path = f"/data1/data_1/anoy/anoy_pipe/LLM/llm_embedding/Meta-Llama-3.1-8B/{style}/lokkagle_diabetes-classification-problem.pt"
        try:
            if os.path.exists(embedding_path):
                emb = torch.load(embedding_path)
                print(emb.dtype, emb.shape)
                embeddings.append(emb)
            else:
                logger.error(f"Embedding file not found: {embedding_path}")
                raise FileNotFoundError(f"Required embedding file missing: {embedding_path}")
        except Exception as e:
            logger.error(f"Error loading embedding from {embedding_path}: {str(e)}")
            raise
        llm_e1=embeddings[0]
    # llm_1,llm_2,llm_3,llm_4= embeddings[0], embeddings[1], embeddings[2], embeddings[3]
    # with torch.no_grad():
    #     fused_embedding = model(llm_1,llm_2,llm_3,llm_4)
    # print(fused_embedding.shape)
    # torch.save(fused_embedding, "fused_embedding.pt")
    print("大模型嵌入 生成 成功",llm_e1.shape)


if __name__ == "__main__":
    main()
