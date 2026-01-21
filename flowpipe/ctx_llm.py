import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger




class LLM_Fused(nn.Module):
    def __init__(self, dim, cross_layer_num):
        super().__init__()

        self.llm_linear = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])  # 4种LLM embedding的独立线性层

    def forward(self, llm_e1, llm_e2, llm_e3, llm_e4):
        # 独立处理每个LLM embedding
        llm_e1 = self.llm_linear[0](llm_e1)
        llm_e2 = self.llm_linear[1](llm_e2)
        llm_e3 = self.llm_linear[2](llm_e3)
        llm_e4 = self.llm_linear[3](llm_e4)

        # 堆叠4种LLM embedding（模拟序列）
        z = torch.stack([llm_e1, llm_e2, llm_e3, llm_e4], dim=1)  # (B, 4, D)
        z = rearrange(z, 'b n d -> b n d')  # 保持形状

        # 多层交叉注意力融合（当前POI_e未启用，可扩展）
        # 假设POI_e是另一个模态的embedding，例如 torch.randn((b, n, d))
        # 这里用占位符模拟
        poi_e = torch.zeros_like(z)  # 临时占位（实际可传入POI embedding）

        for cross_attention_fusion in self.cross_attention_fusion:
            z = cross_attention_fusion(z, poi_e)

        # 最终融合：平均池化
        z = z.mean(dim=1)  # (B, D)

        return z.squeeze()


def load_llm_embedding(llm_name, dataset_name):
    """加载LLM嵌入，独立函数版本"""
    styles = ['pipeline_component', 'raw_data', 'statistical_sampling', 'contextual_semantic']
    embeddings = []

    for style in styles:
        embedding_path = f"LLM/llm_embedding/{llm_name}/{style}/{dataset_name}.pt"
        try:
            if os.path.exists(embedding_path):
                emb = torch.load(embedding_path,weights_only=True)
                embeddings.append(emb)
            else:
                logger.error(f"Embedding file not found: {embedding_path}")
                raise FileNotFoundError(f"Required embedding file missing: {embedding_path}")
        except Exception as e:
            logger.error(f"Error loading embedding from {embedding_path}: {str(e)}")
            raise

    if len(embeddings) != 4:
        raise ValueError(f"Expected 4 embeddings, but loaded {len(embeddings)}")

    return embeddings[0], embeddings[1], embeddings[2], embeddings[3]


def generate_context_embedding(llm_name, dataset_name):
    """生成上下文嵌入：加载并融合4种LLM嵌入"""
    try:
        logger.info(f"Generating context embedding for dataset: {dataset_name}")
        llm_1, llm_2, llm_3, llm_4 = load_llm_embedding(llm_name, dataset_name)
        print("-----------------------------------")
        print(llm_1.shape)
        return llm_1.squeeze()
        # model = LLM_Fused(dim=4096, cross_layer_num=3)
        # ctx_embedding = model(llm_1, llm_2, llm_3, llm_4)
        # print("------------------------------------")
        # print(ctx_embedding.shape)
        # logger.info(f"Context embedding generated successfully: {ctx_embedding.shape}")
        # return ctx_embedding
    except Exception as e:
        logger.error(f"Context embedding generation failed: {str(e)}")
        # 如果LLM嵌入失败，返回一个默认的4096维零向量
        return torch.zeros(4096)


# 测试主函数
if __name__ == "__main__":
    llm_name = "Meta-Llama-3.1-8B"  # 示例LLM名称
    dataset_name = "lokkagle_diabetes-classification-problem"  # 示例数据集名称
    ctx_embedding = generate_context_embedding(llm_name, dataset_name)
    print(ctx_embedding)