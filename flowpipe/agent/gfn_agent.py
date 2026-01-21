import math
import torch
from torch import nn
from torch.distributions import Categorical
from loguru import logger
import os
from config import GFlowNetConfig
from ctxpipe.agent.model import FlowNetwork  # 第 4 步里新增的
from ctxpipe.agent.trajectory_buffer import TrajectoryBuffer
import numpy as np
import comp


class GFNAgent(object):
    def __init__(self, config: GFlowNetConfig):
        self._config = config
        self.device = config.device
        self.model = FlowNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)

        self.failure_weight = getattr(config, "failure_weight", 0.1)
        self.reward_temperature = getattr(config, "reward_temperature", 1.0)
        self.enable_reward_normalization = getattr(config, "enable_reward_normalization", False)
        self.reward_norm_momentum = getattr(config, "reward_norm_momentum", 0.05)

        self._reward_stats = {}
        self._current_ctx = None

        # Initialization logic removed: PyTorch defaults are sufficient, and FiLM layers handle their own init.
        # 消融实验：共享的可学习上下文向量
        self.learnable_ctx = None
        if getattr(config, 'use_learnable_ctx', False):
            # 先在目标设备上创建tensor，再包装成Parameter
            self.learnable_ctx = nn.Parameter(
                torch.randn(config.ctx_dim, device=self.device) * 0.02
            )
            self.optimizer = torch.optim.Adam(
                list(self.model.parameters()) + [self.learnable_ctx],
                lr=config.learning_rate
            )
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.buffer = TrajectoryBuffer(config.buffer_capacity)

    def set_context(self, ctx_tensor):
        if not self._config.use_conditional_gfn:
            self._current_ctx = None
            return
        
        # 消融模式：使用可学习向量
        if getattr(self._config, 'use_learnable_ctx', False) and self.learnable_ctx is not None:
            ctx_tensor = self.learnable_ctx
        
        if ctx_tensor is None:
            ctx_tensor = torch.zeros(self._config.ctx_dim, dtype=torch.float32)
        if ctx_tensor.dim() == 1:
            ctx_tensor = ctx_tensor.unsqueeze(0)
        
        # 消融模式下不detach，需要梯度回传
        if getattr(self._config, 'use_learnable_ctx', False):
            self._current_ctx = ctx_tensor.to(self.device, dtype=torch.float32)
        else:
            self._current_ctx = ctx_tensor.to(self.device, dtype=torch.float32).detach()

    # 平滑 奖励
    # 将整个方法替换为：
    def _normalize_reward(self, dataset_name: str, reward: float) -> float:
        # 仅做数值安全检查，移除所有动态均值方差逻辑
        if math.isnan(reward):
            return 1e-6
        return max(reward, 1e-6)


    def sample_trajectory(self, env, max_steps=None):
        if max_steps is None:
            max_steps = self._config.max_steps
        states, actions, log_pf, masks, errors = [], [], [], [], []

        env_state = env.gfn_get_state()
        done, reward, step_count = False, 1e-6, 0
        last_err = 0

        while not done and step_count < max_steps:
            state_vec = self._encode_state(env_state)
            logits, _ = self.model(state_vec.unsqueeze(0), ctx=self._current_ctx)
            logits = logits.squeeze(0)

            mask_info = env.gfn_get_action_mask()
            masks.append(mask_info["mask"])
            masked_logits = self._apply_mask(logits, mask_info["mask"])
            if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).all():
                last_err = -1
                reward = 1e-6
                break

            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
            action_gid = int(action.item())
            log_pf.append(dist.log_prob(action).item())

            next_state, reward, done, err_code = env.gfn_step(action_gid)
            errors.append(err_code)

            states.append(state_vec.detach().cpu())
            actions.append(action_gid)

            if err_code != 0:
                last_err = err_code
                reward = 1e-6
                break

            env_state = next_state
            step_count += 1

        cached_reward = None
        if hasattr(env, "pipeline") and getattr(env.pipeline, "_eval_cache_valid", False):
            cached_reward = env.pipeline._last_eval_result

        if not done and last_err == 0:
            reward = cached_reward
            if reward is None:
                try:
                    result = env.pipeline.evaluate()
                    if result is not None:
                        reward = float(result)
                except Exception:
                    reward = 1e-6
        else:
            if (reward is None or reward == 0.0) and cached_reward is not None:
                reward = cached_reward

        reward = float(reward if reward is not None else 0.0)
        dataset_name = env._config.classification_task_dic[env.pipeline.taskid]["dataset"]
        reward = self._normalize_reward(dataset_name, reward)
        beta = 1.0 / max(self.reward_temperature, 1e-2)
        reward = reward ** beta
        reward = max(reward, 1e-6)

        if states:
            trajectory = {
                "states": states,
                "actions": actions,
                "log_pf": log_pf,
                "masks": masks,
                "errors": errors,
                "reward": reward,
                # 消融模式下不存ctx，学习时用当前的learnable_ctx
                "ctx": None if getattr(self._config, 'use_learnable_ctx', False) 
                       else (self._current_ctx.detach().cpu() if self._current_ctx is not None else None),

            }
            return reward, trajectory

        return 1e-6, None

    def learn_from_batch(self, trajectories, updates=1, batch_size=8):
        if not trajectories:
            return []

        self.buffer.extend(trajectories)
        losses = []

        for _ in range(updates):
            batch = self.buffer.sample(batch_size)
            if not batch:
                break

            traj_losses = []
            for traj in batch:
                if not traj["states"]:
                    continue

                state_batch = torch.stack(traj["states"]).to(self.device)
                
                # 消融模式：使用当前的可学习向量（需要梯度）
                if getattr(self._config, 'use_learnable_ctx', False) and self.learnable_ctx is not None:
                    ctx_tensor = self.learnable_ctx.unsqueeze(0).expand(state_batch.size(0), -1)
                else:
                    ctx_tensor = traj.get("ctx")
                    if self._config.use_conditional_gfn:
                        if ctx_tensor is None:
                            ctx_tensor = torch.zeros(self._config.ctx_dim, dtype=torch.float32)
                        ctx_tensor = ctx_tensor.to(self.device, dtype=torch.float32)
                        if ctx_tensor.dim() == 1:
                            ctx_tensor = ctx_tensor.unsqueeze(0)
                        ctx_tensor = ctx_tensor.expand(state_batch.size(0), -1)
                    else:
                        ctx_tensor = None
                logits, log_z = self.model(state_batch, ctx=ctx_tensor)

                masks = traj.get("masks", [])
                if masks:
                    mask_tensor = torch.tensor(masks, dtype=logits.dtype, device=logits.device)
                    logits = torch.where(mask_tensor > 0, logits, torch.full_like(logits, float("-inf")))

                actions = torch.tensor(traj["actions"], dtype=torch.long, device=self.device)
                log_probs = torch.log_softmax(logits, dim=-1)
                step_log_pf = log_probs[torch.arange(actions.size(0), device=self.device), actions]
                sum_log_pf = step_log_pf.sum()

                reward_tensor = torch.tensor(traj["reward"], device=self.device).clamp(min=1e-6)
                loss = (sum_log_pf - log_z[0] - torch.log(reward_tensor)) ** 2
                # 失败轨迹 添加 权重
                reward_value = float(traj["reward"])
                is_failure = reward_value <= 1e-6 or any(err != 0 for err in traj.get("errors", []))
                weight = self.failure_weight if is_failure else 1.0
                loss=loss * weight
                traj_losses.append(loss)

            if not traj_losses:
                continue

            batch_loss = torch.stack(traj_losses).mean()
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            losses.append(batch_loss.item())

        return losses


    def learn(self, batch_size=8):
        batch = self.buffer.sample(batch_size)
        return self.learn_from_batch(batch, updates=1, batch_size=batch_size)


    # --- 内部辅助方法 ---
    def _encode_state(self, state_repr):
        logic_flag = np.array(
            [1.0 if state_repr["logic_pipeline_selected"] else 0.0], dtype=np.float32
        )
        logic_choice = np.zeros(comp.num_lpipelines, dtype=np.float32)
        if (
            state_repr["logic_pipeline_selected"]
            and 0 <= state_repr["logic_pipeline_id"] < comp.num_lpipelines
        ):
            logic_choice[state_repr["logic_pipeline_id"]] = 1.0

        stage_onehot = self._one_hot(state_repr["stage_index"], len(comp.logic_pipeline_1))
        predictor_onehot = self._one_hot(state_repr["predictor_id"], comp.num_predictors)

        gid_onehot = np.zeros(self._config.action_dim, dtype=np.float32)
        for gid in state_repr["selected_gids"]:
            if 0 <= gid < self._config.action_dim:
                gid_onehot[gid] = 1.0
        # 3) data_feature -> 转成 np.array，必要时做归一化

        data_feat = np.asarray(state_repr["data_feature"], dtype=np.float32)

        flat = np.concatenate(
            [logic_flag, logic_choice, stage_onehot, predictor_onehot, gid_onehot, data_feat]
        )
        flat = np.nan_to_num(flat, nan=0.0, posinf=1e6, neginf=-1e6)

        tensor = torch.tensor(flat, dtype=torch.float32, device=self.device)
        return tensor


    def _one_hot(self, index, size):
        vec = np.zeros(size, dtype=np.float32)
        if 0 <= index < size:
            vec[index] = 1.0
        return vec

    def _apply_mask(self, logits, mask):
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        mask_tensor = torch.tensor(mask, dtype=logits.dtype, device=logits.device)
        if mask_tensor.sum() <= 0:
            return torch.full_like(logits, float("-inf"))
        masked_logits = torch.where(mask_tensor > 0, logits, torch.full_like(logits, float("-inf")))
        masked_logits = torch.nan_to_num(masked_logits, nan=0.0, posinf=1e6, neginf=-1e6)
        return masked_logits


    def save_model(self, out_dir, tag=""):
        os.makedirs(out_dir, exist_ok=True)
        filename = f"{tag}_gfn.pt" if tag else "gfn.pt"
        path = os.path.join(out_dir, filename)
        save_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self._config.__dict__,
        }
        # 保存可学习上下文向量
        if self.learnable_ctx is not None:
            save_dict["learnable_ctx"] = self.learnable_ctx.data
        torch.save(save_dict, path)

    def load_model(self, out_dir, tag=""):
        filename = f"{tag}_gfn.pt" if tag else "gfn.pt"
        path = os.path.join(out_dir, filename)
        if not os.path.exists(path):
            logger.warning("No checkpoint found at {}", path)
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        # 加载可学习上下文向量
        if self.learnable_ctx is not None and "learnable_ctx" in checkpoint:
            self.learnable_ctx.data = checkpoint["learnable_ctx"]

