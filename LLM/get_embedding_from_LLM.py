#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

# ----------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------

LLM_MODELS = [
    "DeepSeek-R1-0528-Qwen3-8B",
    "DeepSeek-R1-Distill-Llama-8B",
    "Meta-Llama-3.1-8B",
    "Qwen3-8B",
    "gpt2",
    "llama2"
]

DATA_ROOT = "/data1/data_1/anoy/anoy_pipe/LLM/prompt_outputs"
OUTPUT_PATH = "/data1/data_1/anoy/anoy_pipe/LLM/llm_embedding"  # 基础输出路径

PROMPT_STYLES = [
    "contextual_semantic",
    # "statistical_sampling",
    # "raw_data",
    # "pipeline_component"
]

# 模型配置字典 - 区分不同模型的加载方式和配置
MODEL_CONFIGS = {
    "DeepSeek-R1-0528-Qwen3-8B": {
        "model_class": AutoModelForCausalLM,
        "trust_remote_code": True,
        "layer_config_key": "num_hidden_layers",
        "hidden_size_key": "hidden_size"
    },
    "DeepSeek-R1-Distill-Llama-8B": {
        "model_class": AutoModelForCausalLM,
        "trust_remote_code": True,
        "layer_config_key": "num_hidden_layers",
        "hidden_size_key": "hidden_size"
    },
    "Meta-Llama-3.1-8B": {
        "model_class": AutoModelForCausalLM,
        "trust_remote_code": False,
        "layer_config_key": "num_hidden_layers",
        "hidden_size_key": "hidden_size"
    },
    "Qwen3-8B": {
        "model_class": AutoModelForCausalLM,
        "trust_remote_code": True,
        "layer_config_key": "num_layers",
        "hidden_size_key": "hidden_size"
    },
    "gpt2": {
        "model_class": AutoModel,
        "trust_remote_code": False,
        "layer_config_key": "n_layer",
        "hidden_size_key": "n_embd"
    },
    "llama2": {
        "model_class": AutoModel,
        "trust_remote_code": False,
        "layer_config_key": "n_layer",
        "hidden_size_key": "n_embd"
    }
}

SAVE_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16
}

# ----------------------------------------------------------------------
# 参数解析
# ----------------------------------------------------------------------
def create_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="提取 LLM 激活值（最后有效 token 向量）")
    parser.add_argument("--gpu", type=int, default=5, help="选择 GPU 编号（默认 0）")
    parser.add_argument(
        "--LLM",
        type=str,
        default="Meta-Llama-3.1-8B",
        choices=LLM_MODELS,
        help="选择使用的 LLM 模型"
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="all",
        choices=PROMPT_STYLES + ["all"],
        help="选择提示类型，'all' 表示处理所有提示类型"
    )
    parser.add_argument(
        "--target_layer_index",
        type=int,
        default=-1,
        help="要提取的 hidden_states 层索引，-1 表示最后一层，-2 倒数第二层等"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="DataLoader 批大小（根据显存可调）"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="tokenizer 截断最大长度（不超过模型支持上限）"
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="float32",
        choices=list(SAVE_DTYPES.keys()),
        help="保存到磁盘的张量精度（float16/bfloat16/float32）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="deepline",
        choices=["haipipe", "diffprep","deepline"],
        help="选择数据集 (haipipe/diffprep)"
    )
    return parser.parse_args()

# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def get_model_save_path(base_path, model_name, prompt_type, filename):
    """构建模型保存路径: /llm_embedding/模型名称/prompt_style/"""
    model_dir = os.path.join(base_path, model_name, f"{prompt_type}")

    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{filename}.pt")

def infer_torch_dtype_for_infer(device: torch.device):
    """
    推理精度优先级：
    - 如果是 NVIDIA Ampere（8.x）或更高架构，优先 bfloat16；
    - 否则使用 float16；
    - CPU 则 float32。
    """
    if device.type == "cuda":
        return torch.float16
    return torch.float32

def load_model_and_tokenizer(model_name, device):
    """根据模型配置加载 tokenizer 和 模型（按推理 dtype 加载权重，显存友好）"""
    model_path = f"./LLMs/{model_name}"
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    if model_name not in MODEL_CONFIGS:
        logger.error(f"不支持的模型: {model_name}")
        raise ValueError(f"不支持的模型: {model_name}")

    config = MODEL_CONFIGS[model_name]
    logger.info(f"使用配置加载模型 {model_name}: {config}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=config["trust_remote_code"]
    )
    
    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # # 【关键修改】强制从左侧截断，保留 Prompt 末尾的 Summary 和 Sentinel Token
    # tokenizer.truncation_side = 'left'

    # 选择推理 dtype 并加载
    infer_dtype = infer_torch_dtype_for_infer(device)
    model_class = config["model_class"]
    model = model_class.from_pretrained(
        model_path,
        trust_remote_code=config["trust_remote_code"],
        torch_dtype=infer_dtype,
        device_map=None  # 手动 .to(device)
    ).to(device)

    # resize embeddings（仅在设置 pad_token 后可能需要）
    if getattr(model.config, "vocab_size", None) is not None and \
       getattr(tokenizer, "vocab_size", None) is not None and \
       model.config.vocab_size != tokenizer.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    logger.info(f"加载模型 {model_name} 完成，设备: {device}")
    logger.info(f"模型类型: {type(model)}")
    return tokenizer, model

def get_model_layers(model_name, model):
    """根据模型配置获取模型的层数"""
    config = MODEL_CONFIGS[model_name]
    layer_config_key = config["layer_config_key"]

    if hasattr(model.config, layer_config_key):
        num_layers = getattr(model.config, layer_config_key)
        logger.info(f"模型 {model_name} 层数配置键: {layer_config_key}, 层数: {num_layers}")
        return list(range(num_layers))
    else:
        logger.error(f"模型配置中找不到层数配置键: {layer_config_key}")
        raise AttributeError(f"模型配置中找不到层数配置键: {layer_config_key}")

def get_model_hidden_size(model_name, model):
    """根据模型配置获取隐藏层大小"""
    config = MODEL_CONFIGS[model_name]
    hidden_size_key = config["hidden_size_key"]

    if hasattr(model.config, hidden_size_key):
        hidden_size = getattr(model.config, hidden_size_key)
        logger.info(f"模型 {model_name} 隐藏层大小配置键: {hidden_size_key}, 大小: {hidden_size}")
        return hidden_size
    else:
        logger.error(f"模型配置中找不到隐藏层大小配置键: {hidden_size_key}")
        raise AttributeError(f"模型配置中找不到隐藏层大小配置键: {hidden_size_key}")

def process_activation_batch(batch_activations: torch.Tensor,
                             attention_mask: torch.Tensor) -> torch.Tensor:
    """
    从 batch 激活 (B, T, D) 中，提取每条样本最后一个有效 token 的激活向量。
    要求 batch_activations 与 attention_mask 在同一设备。
    """
    last_idx = attention_mask.sum(dim=1) - 1               # (B,)
    last_idx = last_idx.clamp(min=0, max=batch_activations.size(1) - 1)
    gather_index = last_idx.view(-1, 1, 1).expand(-1, 1, batch_activations.size(-1))  # (B,1,D)
    picked = torch.gather(batch_activations, dim=1, index=gather_index).squeeze(1)     # (B,D)
    return picked

# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def process_single_prompt_type(args, device, tokenizer, model, layers, hidden_size, prompt_type):
    """处理单个prompt_type的所有文件：对每个CSV生成 [N, hidden_size] 的张量并保存"""
    logger.info(f"开始处理prompt_type: {prompt_type}")

    prompt_folder = os.path.join(DATA_ROOT, f"{prompt_type}")
    # prompt_folder = os.path.join(DATA_ROOT)

    if not os.path.exists(prompt_folder):
        logger.warning(f"提示文件夹不存在，跳过: {prompt_folder}")
        return

    csv_files = [f for f in os.listdir(prompt_folder) if f.endswith('.csv')]
    logger.info(f"在 {prompt_type} 中找到 {len(csv_files)} 个 CSV 文件")
    if not csv_files:
        logger.warning(f"在 {prompt_type} 中未找到CSV文件，跳过")
        return

    model.eval()  # 明确推理模式

    use_autocast = torch.cuda.is_available()
    # amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8) else torch.float16
    amp_dtype = torch.float16
    # tokenizer 支持的最大序列长度（稳妥起见取 min）
    tokenizer_max_len = getattr(tokenizer, "model_max_length", args.max_length)
    max_len = min(args.max_length, tokenizer_max_len if tokenizer_max_len and tokenizer_max_len > 0 else args.max_length)

    for file in csv_files:
        prompt_data_path = os.path.join(prompt_folder, file)
        try:
            prompt_df = pd.read_csv(prompt_data_path, header=0)
        except Exception as e:
            logger.error(f"加载提示数据失败: {e}")
            continue

        if 'prompt' not in prompt_df.columns:
            logger.error(f"文件缺少 'prompt' 列，跳过: {prompt_type}/{file}")
            continue

        dataset_strings = list(prompt_df['prompt'])
        logger.info(f"处理文件: {prompt_type}/{file}, 行数: {len(prompt_df)}")

        # Tokenization
        tk_result = tokenizer.batch_encode_plus(
            dataset_strings,
            return_tensors='pt',
            padding=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=max_len,
            truncation=True
        )
        token_ids = tk_result['input_ids']
        attention_mask = tk_result['attention_mask']
        logger.info(f"Tokenized input_ids shape: {token_ids.shape}, attention_mask shape: {attention_mask.shape}")

        tk_dataset = Dataset.from_dict({
            'input_ids': token_ids.tolist(),
            'attention_mask': attention_mask.tolist(),
        })
        tk_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

        dataloader = DataLoader(tk_dataset, batch_size=args.batch_size, shuffle=False)
        progress_bar = tqdm(dataloader, desc=f"{prompt_type}/{file}", unit="batch", dynamic_ncols=True)

        # 累计所有样本的最终向量到 CPU，避免占用显存
        per_file_embeddings = []

        with torch.no_grad():
            for step, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attn_mask = batch['attention_mask'].to(device, non_blocking=True)

                if use_autocast and device.type == "cuda":
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                            output_hidden_states=True,
                            use_cache=False
                        )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        output_hidden_states=True,
                        use_cache=False
                    )

                # 选层（默认最后一层）
                try:
                    last_hidden = outputs.hidden_states[args.target_layer_index]  # (B, T, D)
                except Exception as e:
                    logger.error(f"获取 hidden_states 失败: {e}")
                    raise

                picked_batch = process_activation_batch(last_hidden, attn_mask)  # (B, D)

                # 立刻移到 CPU 并按指定精度转换，节省显存与磁盘
                save_dtype = SAVE_DTYPES[args.save_dtype]
                per_file_embeddings.append(picked_batch.detach().to('cpu', dtype=save_dtype))

                # 更新进度
                processed = min((step + 1) * args.batch_size, len(dataset_strings))
                progress_bar.set_postfix({"done": f"{processed}/{len(dataset_strings)}"})

                # 显式释放中间变量（保险）
                del outputs, last_hidden, picked_batch, input_ids, attn_mask

        if len(per_file_embeddings) == 0:
            logger.warning(f"文件无可用样本：{prompt_type}/{file}，跳过保存")
            continue

        # 拼接并保存
        file_tensor = torch.cat(per_file_embeddings, dim=0)   # (N, hidden_size)
        filename_without_ext = os.path.splitext(file)[0]
        save_path = get_model_save_path(OUTPUT_PATH, args.LLM, prompt_type, filename_without_ext)
        torch.save(file_tensor, save_path)
        logger.info(f"保存激活值到: {save_path}; shape={tuple(file_tensor.shape)}, dtype={file_tensor.dtype}")

        # 适度清理显存（文件间清一次即可）
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info(f"完成处理prompt_type: {prompt_type}")

def main():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)
    """主流程：加载模型，提取激活并保存"""
    args = create_args()

    # 设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # [新增] 动态修改全局路径
    global DATA_ROOT, OUTPUT_PATH
    BASE_DIR = "/data1/data_1/anoy/anoy_pipe"

    if args.dataset == "haipipe":
        DATA_ROOT = f"{BASE_DIR}/LLM/prompt_outputs/dataset"
        OUTPUT_PATH = f"{BASE_DIR}/LLM/llm_embedding/dataset"
    elif args.dataset == "diffprep":
        # gen_prompt_final.py 输出到了 LLM/prompt_outputs/diffprep
        DATA_ROOT = f"{BASE_DIR}/LLM/prompt_outputs/diffprep"
        OUTPUT_PATH = f"{BASE_DIR}/LLM/llm_embedding/diffprep"
    elif args.dataset == "deepline":
        print(">>> 正在处理：DiffPrep (测试集)")
        DATA_ROOT = f"{BASE_DIR}/LLM/prompt_outputs/deepline"
        OUTPUT_PATH = f"{BASE_DIR}/LLM/llm_embedding/deepline"

    logger.info(f"当前处理数据集: {args.dataset}")
    logger.info(f"输入路径 (Prompts): {DATA_ROOT}")
    logger.info(f"输出路径 (Embeddings): {OUTPUT_PATH}")

    # 只加载一次模型与tokenizer
    tokenizer, model = load_model_and_tokenizer(args.LLM, device)
    layers = get_model_layers(args.LLM, model)
    hidden_size = get_model_hidden_size(args.LLM, model)
    logger.info(f"模型层数: {len(layers)}, 隐藏层大小: {hidden_size}")

    # 确定要处理的 prompt_types
    if args.prompt_type == "all":
        prompt_types_to_process = PROMPT_STYLES
        logger.info(f"将处理所有prompt_types: {prompt_types_to_process}")
    else:
        prompt_types_to_process = [args.prompt_type]
        logger.info(f"将处理单个prompt_type: {args.prompt_type}")

    # 主循环
    total_prompt_types = len(prompt_types_to_process)
    for i, prompt_type in enumerate(prompt_types_to_process, 1):
        logger.info(f"进度: {i}/{total_prompt_types} - 开始处理 {prompt_type}")
        try:
            process_single_prompt_type(args, device, tokenizer, model, layers, hidden_size, prompt_type)
        except Exception as e:
            logger.error(f"处理 {prompt_type} 时出错: {e}")
            continue

        # 在不同prompt_type之间进行垃圾回收
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info(f"完成 {prompt_type} ({i}/{total_prompt_types})")

    logger.info("所有prompt_types处理完成！")

if __name__ == "__main__":
    # 纯推理任务，关闭梯度更保险
    torch.set_grad_enabled(False)
    main()
