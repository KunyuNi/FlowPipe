    default_agent_config,
    default_env_config,
    default_gfn_config,
    GlobalConfig,  # Note: Added this line
import random
import comp
import torch
from ctxpipe.llm_embedding_loader import llm_embedding_loader
# from ctxpipe.agent.dqn import Agent as DQNAgent
from ctxpipe.agent.gfn_agent import GFNAgent
from ctxpipe.dataset import Dataset
from ctxpipe.env.enviroment import Environment
from ctxpipe.tester import Tester
from ctxpipe.trainer import Trainer, GFNTrainer
import json
import deterministic
from pathlib import Path
from typing import Optional, Tuple
import re
"""DQN manager"""
# class AgentManager:
#     def __init__(self):
#         """
#         data_path: the path of the dataset
#
#         model is saved in information files.
#         """
#         self.agent = Agent(default_agent_config)
#         self.env = Environment(default_env_config, train=False)
#         self.tester = Tester(self.agent, self.env, 0, default_agent_config)
#
#     def train(self, resume_from=0):
#         self.trainer = Trainer(self.agent, self.env, 0, default_agent_config)
#         self.trainer.train(pre_fr=resume_from)
#
#     def inference(self, dataset: Dataset, tag):
#         return self.tester.inference(dataset.path, tag, dataset.name)

# help function

def _find_latest_checkpoint(model_dir: str) -> Tuple[int, Optional[str]]:
    path = Path(model_dir)
    if not path.exists():
        return 0, None

    pattern = re.compile(r"epoch_(\d+)_gfn\.pt$")
    latest_epoch = 0
    latest_tag: Optional[str] = None

    for file in path.iterdir():
        if not file.is_file():
            continue

        match = pattern.match(file.name)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_tag = f"epoch_{epoch}"
        elif file.name == "gfn.pt" and latest_epoch == 0:
            latest_tag = ""

    return latest_epoch, latest_tag

def _load_training_state(model_dir: str):
    state_path = Path(model_dir) / "resume.json"
    if not state_path.exists():
        return None
    try:
        with state_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except (OSError, ValueError) as exc:
        from loguru import logger
        logger.warning("[GFN] failed to load resume.json: {}", exc)
        return None



class AgentManager:
    def __init__(self):
        """
        data_path: the path of the dataset

        model is saved in information files.
        """
        self.env = Environment(default_env_config, train=False)
        self._use_gfn = getattr(default_gfn_config, "use_gfn", False)

        if self._use_gfn:
            self.agent = GFNAgent(default_gfn_config)
            self.tester = None  # GFlowNet doesn't have a corresponding Tester yet
        else:
             # self.agent = DQNAgent(default_agent_config)
             # self.tester = Tester(self.agent, self.env, 0, default_agent_config)
             raise NotImplementedError("DQN agent is not supported in this version")
        # ctx
        self._ctx_cache = {}
    def _get_or_load_ctx(self, dataset_name: str, embedding_types=None):
        cfg = self.config if hasattr(self, "config") else self.agent._config
        if not cfg.use_conditional_gfn:
            return None
        
        # Check for GTE ablation version
        if "gfnc_film_gte" in GlobalConfig.version:
            return self._get_gte_context(dataset_name)

        if embedding_types is None:
            embedding_types = ["contextual_semantic"]  # 需要其他风格时再加

        if dataset_name in self._ctx_cache:
            return self._ctx_cache[dataset_name]  # CPU float32 (ctx_dim,)

        # Ensure loader uses the correct path from config
        from loguru import logger
        if hasattr(cfg, "llm_embedding_path"):
            logger.info(f"DEBUG: cfg.llm_embedding_path = {cfg.llm_embedding_path}")
            if cfg.llm_embedding_path:
                llm_embedding_loader.set_root(cfg.llm_embedding_path)
            else:
                logger.warning("DEBUG: cfg.llm_embedding_path is empty!")
        else:
            logger.warning(f"DEBUG: cfg object {type(cfg)} has no llm_embedding_path attribute!")

        E = llm_embedding_loader.load_all_embeddings(dataset_name, embedding_types)
        if E is None:
            ctx = torch.zeros(cfg.ctx_dim, dtype=torch.float32)
        else:
            ctx = E.to(torch.float32).mean(dim=0)  # (ctx_dim,)

        ctx_cpu = ctx.detach().cpu()
        self._ctx_cache[dataset_name] = ctx_cpu
        return ctx_cpu

    def _get_gte_context(self, dataset_name: str):
        """Ablation: Sample 100 rows and embed using GTE"""
        from ctxpipe.ctx import embedder
        import pandas as pd
        import os
        from loguru import logger
        
        if dataset_name in self._ctx_cache:
            return self._ctx_cache[dataset_name]

        # Construct path pattern: data/dataset_name/data.csv or similar
        # We need to find the actual path. In test_gfn.py we see:
        # data/diffprep_dataset/{dataset_name}/data.csv OR data/deepline_dataset/...
        # This is a bit tricky as AgentManager doesn't inherently know the prefix
        # But we can try the standard locations defined in config/env
        
        # Try finding the dataset file
        possible_prefixes = [
            "data/deepline_dataset",
            "data/diffprep_dataset",
            "data/dataset"
        ]
        
        csv_path = None
        for prefix in possible_prefixes:
            path = os.path.join(prefix, dataset_name, "data.csv")
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            logger.warning(f"Could not find data.csv for GTE ablation for dataset {dataset_name}. Using zero context.")
            return torch.zeros(self.agent._config.ctx_dim, dtype=torch.float32)

        try:
            df = pd.read_csv(csv_path)
            # Sample 100 rows or all if less
            n = min(len(df), 100)
            if n > 0:
                df_sample = df.sample(n=n, random_state=42)
            else:
                df_sample = df
            
            text_data = df_sample.to_string(index=False)
            
            # Embed
            # Note: embedder.embed returns tensor on DEVICE usually, but we cache CPU
            emb = embedder.embed(text_data)
            
            # Ensure it matches ctx_dim
            # GTE typically returns 1024 or 768. Model expects ctx_dim (4096).
            # We might need padding or projection if dims don't match.
            # Assuming for this ablation we just use what it gives, 
            # BUT wait, the model architecture (Linear 4096->128) expects 4096.
            # GTE-large is 1024 dim. 
            # User said "ctx_embedding = embedder.embed(...)".
            # If the user's model expects 4096, and GTE gives 1024, we must pad.
            
            target_dim = self.agent._config.ctx_dim
            current_dim = emb.shape[-1]
            
            if current_dim != target_dim:
                # Pad with zeros
                padding = torch.zeros(target_dim - current_dim, dtype=emb.dtype, device=emb.device)
                emb_padded = torch.cat([emb, padding], dim=-1)
                emb = emb_padded
            
            emb_cpu = emb.detach().cpu()
            self._ctx_cache[dataset_name] = emb_cpu
            return emb_cpu
            
        except Exception as e:
            logger.error(f"Error generating GTE context for {dataset_name}: {e}")
            return torch.zeros(self.agent._config.ctx_dim, dtype=torch.float32)



    def train(self, resume_from=0):
        if self._use_gfn:
            resume_epoch, tag = _find_latest_checkpoint(default_gfn_config.model_dir)
            if tag is not None:
                self.agent.load_model(default_gfn_config.model_dir, tag=tag)
            trainer = GFNTrainer(self.agent, self.env, default_gfn_config)
            state = _load_training_state(default_gfn_config.model_dir)
            if state:
                trainer.load_state(state)
                # 如果状态文件里的 epoch 更高，优先使用
                resume_epoch = max(resume_epoch, int(state.get("epoch", 0)))
            trainer.train(resume_epoch=resume_epoch)
            trainer.train(resume_epoch=resume_epoch)
        else:
            # trainer = Trainer(self.agent, self.env, 0, default_agent_config)
            # trainer.train(pre_fr=resume_from)
            pass

    def inference(self, dataset: Dataset, tag):
        if self._use_gfn:
            return self._gfn_inference(dataset, tag)
        return self.tester.inference(dataset.path, tag, dataset.name)

    def _gfn_inference(self, dataset: Dataset, tag):
        """GFlowNet推理：根据配置选择随机采样或贪婪/Top-N策略"""
        from loguru import logger

        self.agent.load_model(self.agent._config.model_dir, tag=tag)

        if not hasattr(dataset, 'taskid') or dataset.taskid is None:
            raise ValueError('dataset.taskid not provided for GFN inference')
        if not hasattr(dataset, 'predictor') or dataset.predictor is None:
            raise ValueError('dataset.predictor not provided for GFN inference')
        if not hasattr(dataset, 'metric') or dataset.metric is None:
            raise ValueError('dataset.metric not provided for GFN inference')

        mode = getattr(self.agent._config, "inference_mode", "sample")
        top_n = max(1, getattr(self.agent._config, "inference_top_n", 1))

        best_reward = 0.0
        best_sequence = []
        # 推理的时候设置 ctx
        dataset_name = self.env._config.classification_task_dic[dataset.taskid]["dataset"]
        ctx_vec = self._get_or_load_ctx(dataset_name)
        self.agent.set_context(ctx_vec)

        # 随机采样模式：保持原有行为（按策略分布采样 + best-of-N）
        if mode == "sample":
            num_samples = getattr(self.agent._config, 'inference_samples', 10)

            for i in range(num_samples):
                self.env.reset(
                    taskid=dataset.taskid,
                    predictor=dataset.predictor,
                    metric=dataset.metric,
                    default=False,
                )
                max_steps = self.agent._config.max_steps
                reward, _ = self.agent.sample_trajectory(self.env, max_steps=max_steps)

                logic_id = getattr(self.env.pipeline, "logic_pipeline_id", -1)
                sequence = [step.name for step in self.env.pipeline.sequence]

                logger.info(
                    "[GFN][inference] sample={} mode={} logic={} reward={:.6f}",
                    i + 1,
                    mode,
                    logic_id,
                    reward,
                )

                if reward > best_reward:
                    best_reward = reward
                    best_sequence = sequence

        # 贪婪 / Top-N 模式
        elif mode == "greedy":
            max_steps = self.agent._config.max_steps

            # top_n == 1：完全贪婪，多次采样取 best-of-N
            if top_n == 1:
                num_samples = getattr(self.agent._config, 'inference_samples', 10)

                for i in range(num_samples):
                    self.env.reset(
                        taskid=dataset.taskid,
                        predictor=dataset.predictor,
                        metric=dataset.metric,
                        default=False,
                    )
                    reward, _ = self._greedy_sample_trajectory(max_steps, rank=0)

                    logic_id = getattr(self.env.pipeline, "logic_pipeline_id", -1)
                    sequence = [step.name for step in self.env.pipeline.sequence]

                    logger.info(
                        "[GFN][inference] sample={} mode={} logic={} reward={:.6f}",
                        i + 1,
                        mode,
                        logic_id,
                        reward,
                    )

                    if reward > best_reward:
                        best_reward = reward
                        best_sequence = sequence

            # top_n > 1：Top-N 贪婪，对同一 dataset 跑 rank=0..top_n-1 的多条轨迹
            else:
                for rank in range(top_n):
                    self.env.reset(
                        taskid=dataset.taskid,
                        predictor=dataset.predictor,
                        metric=dataset.metric,
                        default=False,
                    )
                    reward, _ = self._greedy_sample_trajectory(max_steps, rank=rank)

                    logic_id = getattr(self.env.pipeline, "logic_pipeline_id", -1)
                    sequence = [step.name for step in self.env.pipeline.sequence]

                    logger.info(
                        "[GFN][inference] mode={} top_n={} rank={} logic={} reward={:.6f}",
                        mode,
                        top_n,
                        rank,
                        logic_id,
                        reward,
                    )

                    if reward > best_reward:
                        best_reward = reward
                        best_sequence = sequence

        else:
            raise ValueError(f"Unsupported inference_mode: {mode}")

        logger.info('GFlowNet inference completed. Best reward: {:.6f}', best_reward)
        return best_sequence, best_reward


    def _greedy_sample_trajectory(self, max_steps, rank: int = 0):
        """贪婪策略采样轨迹"""
        import torch

        states, actions, log_pf, masks, errors = [], [], [], [], []
        env_state = self.env.gfn_get_state()
        done, reward, step_count = False, 1e-6, 0
        last_err = 0

        while not done and step_count < max_steps:
            state_vec = self.agent._encode_state(env_state)
            logits, _ = self.agent.model(state_vec.unsqueeze(0), ctx=self.agent._current_ctx)
            logits = logits.squeeze(0)

            mask_info = self.env.gfn_get_action_mask()
            masks.append(mask_info["mask"])
            masked_logits = self.agent._apply_mask(logits, mask_info["mask"])
            # 贪婪/Top-k 选择：按 logits 排名第 rank 的动作
            k = rank + 1
            values, indices = torch.topk(masked_logits, k=k)
            # 如果合法动作数小于 rank+1，就退到最后一个合法动作
            idx = indices[rank if rank < indices.numel() else indices.numel() - 1]
            action_gid = int(idx.item())

            # 记录log概率（用于可能的后续分析）
            dist = torch.distributions.Categorical(logits=masked_logits)
            log_pf.append(dist.log_prob(torch.tensor([action_gid], device=masked_logits.device)).item())

            next_state, reward, done, err_code = self.env.gfn_step(action_gid)
            errors.append(err_code)

            states.append(state_vec.detach().cpu())
            actions.append(action_gid)

            if err_code != 0:
                last_err = err_code
                reward = 1e-6
                break

            env_state = next_state
            step_count += 1

        # 获取最终奖励
        cached_reward = None
        if hasattr(self.env, "pipeline") and getattr(self.env.pipeline, "_eval_cache_valid", False):
            cached_reward = self.env.pipeline._last_eval_result

        if not done and last_err == 0:
            reward = cached_reward
            if reward is None:
                try:
                    result = self.env.pipeline.evaluate()
                    if result is not None:
                        reward = float(result)
                except Exception:
                    reward = 1e-6

        reward = float(reward if reward is not None else 0.0)
        dataset_name = self.env._config.classification_task_dic[self.env.pipeline.taskid]["dataset"]
        reward = self.agent._normalize_reward(dataset_name, reward)
        reward /= max(self.agent.reward_temperature, 1e-6)
        reward = max(reward, 1e-6)

        if states:
            trajectory = {
                "states": states,
                "actions": actions,
                "log_pf": log_pf,
                "masks": masks,
                "errors": errors,
                "reward": reward,
            }
            return reward, trajectory

        return 1e-6, None
