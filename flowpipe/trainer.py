import gc
import math
import os
from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union
import torch
import numpy as np
import pandas as pd
from loguru import logger
import time
import comp
import json
from config import AgentConfig
from config import default_gfn_config
from util import abspath
import util  # 复用已有工具
import random
import deterministic
from ctxpipe.agent.trajectory_buffer import TrajectoryBuffer

from ctxpipe.agent.dqn import Agent
from ctxpipe.stats import Stats
from ctxpipe.ctx import TableEmbedder, TextEmbedder, embedder
from ctxpipe.env.enviroment import Environment
from ctxpipe.env.primitives.imputercat import ImputerCatPrim
from ctxpipe.env.primitives.primitive import Primitive
from ctxpipe.ctx_llm import generate_context_embedding
# from flowpipe.llm_embedding_fusion import LLM_Fused
from sklearn.cluster import KMeans

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Trainer:

    def __init__(self, agent: Agent, env: Environment, test_pred, config: AgentConfig):
        self.agent = agent
        self.agent.no_random = False

        self.env = env
        self._config = config
        self.test_pred = test_pred

        self.imputernum_state = None
        self.imputernum_action = None

        self.epsilon_start = self._config.epsilon_start
        self.epsilon_final = self._config.epsilon_min
        self.epsilon_decay = self._config.eps_decay

        self.outputdir = self._config.model_dir

    def step(
        self,
        fr: int,
        state: np.ndarray,
        one_pip_ismodel: list,
        seq: list,
    ):
        epsilon = self._epsilon_by_frame(fr)
        logger.debug(f"epsilon={epsilon}")
        pipeline_index = self.env.pipeline.get_index()
        has_num_nan, has_cat_nan = self.env.has_nan()
        tried_list = []
        repeat_time=0
        action = -1
        step: Primitive = Primitive()
        reward = -1

        try:
            # 1024 维度
            # ctx_embedding = embedder.embed(self._ctx_data(embedder=embedder))  # type: ignore
            print("--------------------------start")
            '''
            # llm_1,llm_2,llm_3,llm_4=self._load_llm_embedding()
            # model = LLM_Fused(dim=4096, cross_layer_num=3)
            # ctx_embedding=model(llm_1,llm_2,llm_3,llm_4)
            '''
            ctx_embedding = generate_context_embedding(self._config.LLM_name, self.curr_dataset)
            print("大模型嵌入融合生成成功",ctx_embedding.shape)
            # dataset_name = self.curr_dataset  # 从 classification_task_dic.json 获取的 dataset name
            # styles = ['pipeline_component', 'raw_data', 'statistical_sampling', 'contextual_semantic']  # 四种固定风格
            # embeddings = []
            # for style in styles:
            #     embedding_path = f"LLM/llm_embedding/{self._config.LLM_name}/{style}/{dataset_name}.pt" # 大模型名称暂时固定
            #     print(embedding_path)
            #     if os.path.exists(embedding_path):
            #         emb = torch.load(embedding_path)
            #         embeddings.append(emb)
            #     else:
            #         logger.warning(f"Embedding file not found: {embedding_path}, skipping")
            # if embeddings:
            #     ctx_embedding = torch.mean(torch.stack(embeddings), dim=0)  # 平均四个 embedding
            # else:
            #     raise ValueError(f"No valid embeddings found for dataset: {dataset_name}")
            # ctx_embedding = embedder.embed(self._ctx_data(embedder=embedder))  # 原逻辑，注释掉

        except:
            logger.error("Context embedding error")
            state, one_pip_ismodel, seq = self._reset(epsilon=epsilon)
            return state, one_pip_ismodel, seq

        """select an action by epsilon-greedy"""
        logic_pipeline_id = self.env.pipeline.logic_pipeline_id
        curr_component = comp.lpipelines[logic_pipeline_id][pipeline_index]

        logger.debug(f"Current component: {curr_component}")

        step_result: Optional[List] = None
        while True:
            if curr_component == "ImputerNum":
                action, step = self._act_imputernum(
                    ctx_embedding, has_num_nan, state, tried_list, epsilon
                )
            elif curr_component == "ImputerCat":
                action, step = self._act_imputercat(has_cat_nan)
            elif curr_component == "Encoder":
                action, step = self._act_encoder(state, tried_list, epsilon)
            elif curr_component == "FeaturePreprocessing":
                action, step = self._act_feature_preprocessing(
                    state, tried_list, epsilon
                )
            elif curr_component == "FeatureEngine":
                action, step = self._act_feature_engine(state, tried_list, epsilon)
            elif curr_component == "FeatureSelection":
                action, step = self._act_feature_selection(state, tried_list, epsilon)
            else:
                raise ValueError(f"No such component type: {curr_component}")

            if action in tried_list:
                repeat_time += 1
                continue

            tried_list.append(action)

            """execute the action"""
            step_result, err = self.env.step(step)

            if err is None:
                logger.debug(f"[{curr_component}] ACT {step.name}")
                break

            if err is not None and len(tried_list) > 0:
                step_result = None
                reward = err
                logger.warning(f"Exceeded max retry time. Reward: {reward}")
                break

            logger.warning(
                f"Retried [{len(tried_list)}] for {curr_component}: {tried_list}"
            )

        if step_result is None:
            is_pipeline_done = True
        else:
            state, reward, next_state, is_pipeline_done = step_result
            seq.append(action)

            if curr_component == "ImputerNum":
                self.imputernum_state = state
                self.imputernum_action = action
            elif curr_component == "ImputerCat":
                self.agent.buffer.add(
                    self.imputernum_state,
                    self.imputernum_action,
                    reward,
                    next_state,
                    False,
                    "ImputerNum",
                    logic_pipeline_id,
                    ctx_embedding,
                )
            else:
                self.agent.buffer.add(
                    state,
                    action,
                    reward,
                    next_state,
                    is_pipeline_done,
                    curr_component,
                    logic_pipeline_id,
                    ctx_embedding,
                )

            state = next_state

            """save checkpoint"""
            if fr % self._config.checkpoint_interval == 0:
                self.agent.save_model(self.outputdir, tag=f"ctx_{fr}")

            """update model"""
            logger.debug(
                f"buffer_size={self.agent.buffer.size()}, lp_size={self.agent.buffer.lp_size()}"
            )
            if (
                self.agent.buffer.size() >= self._config.batch_size
                and self.agent.buffer.lp_size() >= self._config.logic_batch_size
                and fr % self._config.backpropagate_interval == 0
            ):
                self.agent.learn_components()
                self.agent.learn_lp()
                self._result_log["learn_time"] += 1

        if is_pipeline_done:
            logger.info(
                f"Dataset: {self.curr_dataset}. "
                f"Pipeline: {self.env.pipeline.sequence}. "
                f"Predictor: {self.env.pipeline.predictor.name}. "
                f"Reward: {reward}"
            )

            if step_result is not None:
                """add sample"""
                self.agent.buffer.lp_add(
                    self.env.lpip_state,
                    logic_pipeline_id,
                    reward,
                    self.agent.last_raw_dataset_ctx,
                )

                # self._log_pipeline(one_pip_ismodel, reward, seq)

            state, one_pip_ismodel, seq = self._reset(epsilon)

        return state, one_pip_ismodel, seq

    def _reset(self, epsilon: float):
        self.env.reset()

        self.env.pipeline.logic_pipeline_id, _ = self.agent.act(
            self.env.pipeline, self.env.lpip_state, "LogicPipeline", epsilon=epsilon
        )

        state = self.env.get_state()
        one_pip_ismodel = []
        seq = []

        return state, one_pip_ismodel, seq

    @property
    def curr_dataset(self):
        return self._config.classification_task_dic[self.env.pipeline.taskid]["dataset"]

    def train(self, pre_fr=0):
        one_pip_ismodel = []
        seq = []

        self._init_log()
        if pre_fr>0:
            self.agent.load_weights(self.outputdir, tag=f"ctx_{pre_fr}")
        else:
            self.agent.save_model(self.outputdir, tag=f"ctx_0")

        self.env.reset() # 重置环境
        self.env.pipeline.logic_pipeline_id, _ = self.agent.act(
            self.env.pipeline,
            self.env.lpip_state,
            "LogicPipeline",
            epsilon=self._epsilon_by_frame(0),
        )

        state = self.env.get_state()

        for fr in range(pre_fr + 1, self._config.frames + 1):
            state, one_pip_ismodel, seq = self.step(fr, state, one_pip_ismodel, seq)

            logger.info(
                f"[TRAIN] Step {fr} - dataset: {self.curr_dataset} "
                f"seq: {self.env.pipeline.sequence} "
                f"model: {self.env.pipeline.predictor.name}"
            )

            gc.collect()

    def _epsilon_by_frame(self, frame_id):
        return self.epsilon_final + (
            self.epsilon_start - self.epsilon_final
        ) * math.exp(-1.0 * frame_id / self.epsilon_decay)
    """old"""
    def _ctx_data(self, embedder: Union[TextEmbedder, TableEmbedder]):
        if isinstance(embedder, TextEmbedder):
            if len(self.env.pipeline.train_x.index) > 100:
                result = self.env.pipeline.train_x.sample(n=100).to_csv(index=False)# 100行
            else:
                result = self.env.pipeline.train_x.to_csv(index=False)
        elif isinstance(embedder, TableEmbedder):
            result = self.env.pipeline.train_x.sample(n=embedder.MAX_N_ROWS)
            if len(result.columns) > embedder.MAX_N_COLUMNS:
                result = result.iloc[:, -embedder.MAX_N_COLUMNS :]
        return result #2831
    """old"""
    def _ctx_data_new(self, embedder: Union['TextEmbedder', 'TableEmbedder']):
        df = self.env.pipeline.train_x
        n_samples = 100
        n_clusters = 8
        RANDOM_SEED = 0

        if isinstance(embedder, TextEmbedder):
            if len(df) > n_samples:
                # 只用数值特征聚类，聚类前先去除缺失
                X = df.select_dtypes(include='number').dropna()
                valid_idx = X.index
                X = X.reset_index(drop=True)

                # 核心健壮性判断：无特征、样本数太少、全常数
                if (
                    X.shape[1] == 0
                    or len(X) < n_clusters
                    or (X.nunique() <= 1).all()
                ):
                    sampled_df = df.sample(n=n_samples, random_state=RANDOM_SEED)
                else:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init='auto')
                    clusters = kmeans.fit_predict(X)
                    centers = kmeans.cluster_centers_

                    samples_per_cluster = [n_samples // n_clusters] * n_clusters
                    for i in range(n_samples % n_clusters):
                        samples_per_cluster[i] += 1

                    selected_idx = []
                    for i in range(n_clusters):
                        cluster_idx = np.where(clusters == i)[0]
                        if len(cluster_idx) == 0:
                            continue
                        cluster_points = X.iloc[cluster_idx]
                        center = centers[i]
                        distances = ((cluster_points - center) ** 2).sum(axis=1)
                        repr_pos = distances.idxmin()
                        repr_idx = valid_idx[repr_pos]  # 转回原df的index
                        selected = {repr_idx}
                        n_in_cluster = samples_per_cluster[i] - 1
                        if n_in_cluster > 0:
                            others = list(set(valid_idx[cluster_points.index]) - {repr_idx})
                            if len(others) > 0:
                                n_actual = min(n_in_cluster, len(others))
                                add_idx = np.random.RandomState(RANDOM_SEED).choice(
                                    others, size=n_actual, replace=False
                                )
                                selected.update(add_idx)
                        selected_idx.extend(selected)
                        del cluster_idx, cluster_points, center, distances
                        gc.collect()

                    # 补齐采样数量
                    if len(selected_idx) < n_samples:
                        rest_needed = n_samples - len(selected_idx)
                        available_idx = list(set(df.index) - set(selected_idx))
                        if rest_needed <= len(available_idx):
                            fill_idx = np.random.RandomState(RANDOM_SEED).choice(
                                available_idx, size=rest_needed, replace=False
                            )
                            selected_idx.extend(fill_idx)
                        else:
                            selected_idx.extend(available_idx)

                    # 保证唯一和顺序
                    seen = set()
                    ordered_idx = []
                    for idx in selected_idx:
                        if idx not in seen:
                            ordered_idx.append(idx)
                            seen.add(idx)
                    ordered_idx = ordered_idx[:n_samples]

                    del kmeans, clusters, centers, X, selected_idx, seen
                    gc.collect()

                    sampled_df = df.loc[ordered_idx].reset_index(drop=True)
            else:
                sampled_df = df.copy()
            result = sampled_df.to_csv(index=False)

        elif isinstance(embedder, TableEmbedder):
            sampled_df = df.sample(n=min(len(df), embedder.MAX_N_ROWS), random_state=RANDOM_SEED)
            if len(sampled_df.columns) > embedder.MAX_N_COLUMNS:
                sampled_df = sampled_df.iloc[:, -embedder.MAX_N_COLUMNS :]
            # result = sampled_df.to_csv(index=False)
        else:
            sampled_df = df.head(n_samples)
            result = sampled_df.to_csv(index=False)
        del df
        gc.collect()
        return result



    def _do_agent_act(
        self, state, component_type, tried_list, epsilon
    ) -> Tuple[int, bool]:
        action, is_model = self.agent.act(
            self.env.pipeline, state, component_type, tried_list, epsilon
        )

        return action, is_model

    def _log_action(self, component_type: str, step: Primitive):
        pass
        # with open("action.temp.log", "a") as f:
        #     f.write(f"{component_type} {step.name}\n")

    def _act_imputernum(
        self, ctx_embedding, has_num_nan, state, tried_list, epsilon
    ) -> Tuple[int, Primitive]:
        self.agent.set_last_raw_dataset_ctx(ctx_embedding)

        if not has_num_nan:
            return len(comp.imputernums), Primitive()

        action, is_model = self._do_agent_act(state, "ImputerNum", tried_list, epsilon)
        step: Primitive = deepcopy(comp.imputernums[action])

        if is_model:
            self._log_action("imputernum", step)

        return action, step

    def _act_imputercat(self, has_cat_nan) -> Tuple[int, Primitive]:
        if not has_cat_nan:
            return -1, Primitive()

        return -1, ImputerCatPrim()

    def _act_encoder(self, state, tried_list, epsilon) -> Tuple[int, Primitive]:
        if not self.env.has_cat_cols():
            return len(comp.encoders), Primitive()

        action, is_model = self._do_agent_act(state, "Encoder", tried_list, epsilon)
        step = deepcopy(comp.encoders[action])

        if is_model:
            self._log_action("encoder", step)

        return action, step

    def _act_feature_preprocessing(
        self, state, tried_list, epsilon
    ) -> Tuple[int, Primitive]:
        action, is_model = self._do_agent_act(
            state, "FeaturePreprocessing", tried_list, epsilon
        )
        step = deepcopy(comp.fpreprocessings[action])

        if is_model:
            self._log_action("featurepreprocessing", step)

        return action, step

    def _act_feature_engine(self, state, tried_list, epsilon) -> Tuple[int, Primitive]:
        action, is_model = self._do_agent_act(
            state, "FeatureEngine", tried_list, epsilon
        )
        step = deepcopy(comp.fengines[action])

        if is_model:
            self._log_action("featureengine", step)

        return action, step

    def _act_feature_selection(
        self, state, tried_list, epsilon
    ) -> Tuple[int, Primitive]:
        action, is_model = self._do_agent_act(
            state, "FeatureSelection", tried_list, epsilon
        )
        step = deepcopy(comp.fselections[action])

        if is_model:
            self._log_action("featureselection", step)

        return action, step

    def _init_log(self):
        self._result_log = {}
        if os.path.exists(self._config.result_log_file_name):
            self._result_log: Any = np.load(
                self._config.result_log_file_name, allow_pickle=True
            ).item()

        for k in ["reward_dic", "max_action", "max_reward", "seq_log"]:
            if k not in self._result_log:
                self._result_log[k] = {}
        if "learn_time" not in self._result_log:
            self._result_log["learn_time"] = 0

    def _log_pipeline(self, one_pip_ismodel, reward, seq):
        for i in range(len(one_pip_ismodel)):
            if one_pip_ismodel[i] == True:
                if self.env.pipeline.taskid not in self._result_log["max_reward"]:
                    self._result_log["max_reward"][self.env.pipeline.taskid] = {}

                if i not in self._result_log["max_reward"][self.env.pipeline.taskid]:
                    self._result_log["max_reward"][self.env.pipeline.taskid][i] = []

                if i not in self._result_log["max_action"]:
                    self._result_log["max_action"][i] = []

                self._result_log["max_reward"][self.env.pipeline.taskid][i].append(
                    reward
                )
                self._result_log["max_action"][i].append(seq[4])

        if self.env.pipeline.taskid not in self._result_log["seq_log"]:
            self._result_log["seq_log"][self.env.pipeline.taskid] = []

        self._result_log["seq_log"][self.env.pipeline.taskid].append(
            (
                self._result_log["learn_time"],
                [i.name for i in self.env.pipeline.sequence],
                reward,
                one_pip_ismodel,
            )
        )

        if self.env.pipeline.taskid not in self._result_log["reward_dic"]:
            self._result_log["reward_dic"][self.env.pipeline.taskid] = []

        self._result_log["reward_dic"][self.env.pipeline.taskid].append(reward)
        np.save(self._config.result_log_file_name, self._result_log)
        # self.agent.save_model(self.outputdir, 'best')


    # def _load_llm_embedding(self):
    #     dataset_name = self.curr_dataset
    #     styles = ['pipeline_component', 'raw_data', 'statistical_sampling', 'contextual_semantic']
    #     embeddings = []
    #
    #     for style in styles:
    #         embedding_path = f"LLM/llm_embedding/{self._config.LLM_name}/{style}/{dataset_name}.pt"
    #         try:
    #             if os.path.exists(embedding_path):
    #                 emb = torch.load(embedding_path)  # 如需跨设备可：map_location='cpu'
    #                 embeddings.append(emb)
    #             else:
    #                 logger.error(f"Embedding file not found: {embedding_path}")
    #                 raise FileNotFoundError(f"Required embedding file missing: {embedding_path}")
    #         except Exception as e:
    #             logger.error(f"Error loading embedding from {embedding_path}: {str(e)}")
    #             raise
    #
    #     if len(embeddings) != 4:
    #         raise ValueError(f"Expected 4 embeddings, but loaded {len(embeddings)}")
    #
    #     return embeddings[0], embeddings[1], embeddings[2], embeddings[3]


    # def _ctx_data_fault(self, embedder: Union[TextEmbedder, TableEmbedder]):
    #     df = self.env.pipeline.train_x
    #     n_samples = 100
    #     if isinstance(embedder, TextEmbedder):
    #         if len(df) > n_samples:
    #             # 只用数值型特征做聚类，非数值型可先做编码
    #             numeric_df = df.select_dtypes(include='number')
    #             if numeric_df.shape[1] == 0:
    #                 # 如果没有数值特征，退回原始随机采样
    #                 result = df.sample(n=n_samples).to_csv(index=False)
    #             else:
    #                 kmeans = KMeans(n_clusters=n_samples, random_state=0)
    #                 clusters = kmeans.fit_predict(numeric_df)
    #                 # 每个簇选距离中心最近的样本
    #                 centers = kmeans.cluster_centers_
    #                 selected_idx = []
    #                 for i in range(n_samples):
    #                     cluster_points = numeric_df[clusters == i]
    #                     if len(cluster_points) == 0:
    #                         continue
    #                     center = centers[i]
    #                     distances = ((cluster_points - center) ** 2).sum(axis=1)
    #                     idx = distances.idxmin()
    #                     selected_idx.append(idx)
    #                 result = df.loc[selected_idx].to_csv(index=False)
    #         else:
    #             result = df.to_csv(index=False)
    #     elif isinstance(embedder, TableEmbedder):
    #         # 这里同理可做代表性采样
    #         result = df.sample(n=min(len(df), embedder.MAX_N_ROWS))
    #         if len(result.columns) > embedder.MAX_N_COLUMNS:
    #             result = result.iloc[:, -embedder.MAX_N_COLUMNS :]
    #     return result #2883


class GFNTrainer:
    def __init__(self, agent, env, config=default_gfn_config):
        self.agent = agent
        self.env = env
        self.config = config
        self.best_reward_per_dataset = {}
        self.iteration = 0
        self.output_dir = config.model_dir
        self._traj_pool = []

        os.makedirs(self.output_dir, exist_ok=True)

        self._metric = self._get_metric()
        self._all_combos = self._build_training_combos()
        self._combo_rng = self._create_combo_rng()
        if self._all_combos:
            self._combo_rng.shuffle(self._all_combos)
        #ctx
        self._ctx_cache={}

    def _get_or_load_ctx(self, dataset_name: str, embedding_types=None):
        if not self.config.use_conditional_gfn:
            return None
        if embedding_types is None:
            embedding_types = ["contextual_semantic"]
        if dataset_name in self._ctx_cache:
            return self._ctx_cache[dataset_name]

        from ctxpipe.llm_embedding_loader import llm_embedding_loader
        import torch
        E = llm_embedding_loader.load_all_embeddings(dataset_name, embedding_types)
        if E is None:
            ctx = torch.zeros(self.config.ctx_dim, dtype=torch.float32)
        else:
            ctx = E.to(torch.float32).mean(dim=0)

        ctx_cpu = ctx.detach().cpu()
        self._ctx_cache[dataset_name] = ctx_cpu
        return ctx_cpu

    def _state_file_path(self) -> str:
        return os.path.join(self.output_dir, "resume.json")

    def load_state(self, state: dict) -> None:
        if not state:
            return
        self.best_reward_per_dataset = state.get("best_per_dataset", {}) or {}
        self.iteration = int(state.get("epoch", 0))
        self.combo_counter = int(state.get("combo_counter", 0))  # 恢复combo计数器
        # 兼容旧文件，防止缺字段
        global_best = state.get("global_best")
        if global_best is not None:
            for dataset, reward in self.best_reward_per_dataset.items():
                if reward == global_best:
                    break

    def _save_training_state(self, epoch: int,combo_counter: int = 0) -> None:
        state = {
            "epoch": epoch,
            "combo_counter": combo_counter,  # 添加combo计数器

            "global_best": self._global_best_reward(),
            "best_per_dataset": self.best_reward_per_dataset,
        }
        state_path = self._state_file_path()
        tmp_path = state_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f)
            os.replace(tmp_path, state_path)
        except OSError as exc:
            logger.warning("[GFN] failed to persist training state: {}", exc)

    def _reset_buffer(self):
        self.agent.buffer = TrajectoryBuffer(self.config.buffer_capacity)
        self._traj_pool = []

    def _get_dataset_name(self, taskid):
        key = str(taskid)
        info = self.env._config.classification_task_dic.get(key)
        if info is None:
            info = self.env._config.classification_task_dic.get(taskid)
        if isinstance(info, dict) and 'dataset' in info:
            return info['dataset']
        return key

    def _update_best_reward(self, dataset_name, reward):
        if reward is None:
            return
        current = self.best_reward_per_dataset.get(dataset_name, 0.0)
        if reward > current:
            self.best_reward_per_dataset[dataset_name] = reward

    def _global_best_reward(self):
        return max(self.best_reward_per_dataset.values()) if self.best_reward_per_dataset else 0.0

    def _create_combo_rng(self):
        seed = getattr(self.config, "coverage_seed", None)
        if seed is None:
            seed = getattr(deterministic, "RANDOM_SEED", None)
        return random.Random(seed)

    def _select_epoch_combos(self, epoch_index, combos_per_epoch, coverage_epochs):
        if not self._all_combos or combos_per_epoch <= 0:
            return []
        if epoch_index < coverage_epochs:
            start = epoch_index * combos_per_epoch
            end = min(start + combos_per_epoch, len(self._all_combos))
            return self._all_combos[start:end]
        sample_size = min(combos_per_epoch, len(self._all_combos))
        if sample_size == 0:
            return []
        return self._combo_rng.sample(self._all_combos, sample_size)


    def train(self, num_epochs=None, resume_epoch=0):
        combos_per_epoch = getattr(self.config, "combos_per_epoch", len(self._all_combos))
        traj_per_combo = getattr(self.config, "traj_per_combo", 8)
        updates_per_combo = getattr(self.config, "updates_per_combo", 1)
        batch_size = getattr(self.config, "trajectory_batch_size", 8)
        combo_counter = getattr(self, 'combo_counter', 0)  # 从对象状态恢复，或默认为0

        if not self._all_combos:
            logger.warning("No training combos available, skip GFN training.")
            return

        total_combos = len(self._all_combos)
        if combos_per_epoch <= 0:
            combos_per_epoch = total_combos or 1

        coverage_epochs = math.ceil(total_combos / combos_per_epoch)
        if num_epochs is None:
            total_epochs = coverage_epochs
            if getattr(self.config, "num_epochs", None) != coverage_epochs:
                logger.info(
                    "[GFN] setting config.num_epochs to {} to cover {} combos ({} per epoch).",
                    coverage_epochs,
                    total_combos,
                    combos_per_epoch,
                )
            self.config.num_epochs = coverage_epochs
        else:
            total_epochs = max(num_epochs, coverage_epochs)
            if total_epochs > num_epochs:
                logger.info(
                    "[GFN] extending epochs to {} to ensure coverage of {} combos ({} per epoch).",
                    total_epochs,
                    total_combos,
                    combos_per_epoch,
                )
            self.config.num_epochs = total_epochs
        self._coverage_epochs = coverage_epochs

        if resume_epoch >= total_epochs:
            logger.info(
                "[GFN] resume_epoch {} >= total_epochs {}, nothing to train.",
                resume_epoch,
                total_epochs,
            )
            return

        start_epoch = max(resume_epoch + 1, 1)
        self.iteration = resume_epoch  # 保持迭代计数连续

        logger.info(
            "Start GFlowNet training: total_epochs={} start_epoch={} coverage_epochs={} combos/epoch={} "
            "traj/combo={} updates/combo={}",
            total_epochs,
            start_epoch,
            coverage_epochs,
            combos_per_epoch,
            traj_per_combo,
            updates_per_combo,
        )

        #
        for epoch in range(start_epoch, total_epochs + 1):
            self.iteration = epoch
            start = time.time()
            active_combos = self._select_epoch_combos(epoch - 1, combos_per_epoch, coverage_epochs)
            if not active_combos:
                logger.warning("[GFN] epoch {} has no combos to train after scheduling.", epoch)
                continue

            epoch_rewards = []
            epoch_losses = []

            for combo in active_combos:
                taskid, predictor, metric = combo
                dataset_name = self._get_dataset_name(taskid)
                combo_rewards = []
                # self._reset_buffer()

                for _ in range(traj_per_combo):
                    reward = self._rollout_combo(taskid, predictor, metric)
                    if reward is None:
                        continue
                    combo_rewards.append(reward)
                    epoch_rewards.append(reward)
                    self._update_best_reward(dataset_name, reward)

                if not combo_rewards:
                    logger.warning(
                        "[GFN] combo(task={}, predictor={}) produced no valid trajectories",
                        taskid,
                        predictor.name if hasattr(predictor, "name") else predictor.__class__.__name__,
                    )
                    continue

                combo_losses = self._optimize_combo_batch(updates_per_combo, batch_size)
                if combo_losses:
                    epoch_losses.extend(combo_losses)

                combo_reward = sum(combo_rewards) / len(combo_rewards)
                predictor_name = predictor.name if hasattr(predictor, "name") else predictor.__class__.__name__
                best_dataset_reward = self.best_reward_per_dataset.get(dataset_name, 0.0)
                combo_counter+=1
                if combo_counter % 10 == 0:
                    self.agent.save_model(self.output_dir, tag=f"combo_{combo_counter}")
                    self._save_training_state(epoch, combo_counter)
                    logger.info(f"[GFN] Saved checkpoint at combo {combo_counter}")

                logic_id = getattr(self.env.pipeline, "logic_pipeline_id", -1)
                logger.info(
                    "[GFN][epoch={}] combo(task={}, predictor={}, logic={}) reward_mean={:.6f} best_dataset={:.6f}",
                    epoch,
                    taskid,
                    predictor_name,
                    logic_id,
                    combo_reward,
                    best_dataset_reward,
                )

            elapsed = time.time() - start
            avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else None

            avg_loss_str = "None" if avg_loss is None else f"{avg_loss:.4f}"
            global_best = self._global_best_reward()
            logger.info(
                "[GFN] epoch={} combo_count={} reward_mean={:.6f} best_global={:.6f} "
                "loss={} elapsed={:.1f}s",
                epoch,
                len(active_combos),
                avg_reward,
                global_best,
                avg_loss_str,
                elapsed,
            )

            self._log_stats(epoch, avg_reward, avg_loss, elapsed)

            # if epoch % self.config.checkpoint_interval == 0:
            #     self.agent.save_model(self.output_dir, tag=f"epoch_{epoch}")
            #     self._save_training_state(epoch,combo_counter)

        self._save_training_state(total_epochs,combo_counter)
        logger.info(
            "GFlowNet training finished at epoch {}. Best reward = {:.6f}",
            total_epochs,
            self._global_best_reward(),
        )


    def _log_stats(self, epoch, avg_reward, avg_loss, elapsed):
        loss_str = "None" if avg_loss is None else f"{avg_loss:.4f}"
        global_best = self._global_best_reward()
        logger.debug(
            f"[GFN][stats] epoch={epoch} reward_mean={avg_reward:.6f} "
            f"best_global={global_best:.6f} loss={loss_str} elapsed={elapsed:.1f}s"
        )
    # --- helpers -------------------------------------------------------------
    def _get_metric(self):
        metric_id = self.env._config.classification_metric_id
        for metric in comp.metrics:
            if metric.id == metric_id:
                return metric
        raise ValueError(f"Metric id {metric_id} not found in comp.metrics")

    def _build_training_combos(self):
        combos = []
        for taskid in self.env._config.train_index:
            for predictor in comp.predictors:
                combos.append((taskid, predictor, self._metric))
        return combos
    def _rollout_combo(self, taskid, predictor, metric):
        try:
            self.env.reset_to_combo(taskid, predictor, metric)
        except Exception as exc:
            logger.warning(
                "[GFN] reset_to_combo failed: task={} predictor={} err={}",
                taskid,
                predictor.name if hasattr(predictor, "name") else predictor.__class__.__name__,
                exc,
            )
            return None

        dataset_name = self.env._config.classification_task_dic[taskid]["dataset"]
        ctx_vec = self._get_or_load_ctx(dataset_name)
        self.agent.set_context(ctx_vec)

        logger.debug(
            "[GFN] rollout task={} dataset={} predictor={}",
            taskid,
            dataset_name,
            predictor.name if hasattr(predictor, "name") else predictor.__class__.__name__,
        )

        reward, trajectory = self.agent.sample_trajectory(
            self.env, max_steps=self.config.max_steps
        )

        if trajectory is not None:
            # 如需让 agent 的全局 buffer 也保留，可取消下面这一行的注释
            # self.agent.buffer.add(trajectory)
            self._traj_pool.append(trajectory)
        return reward



    def _optimize_combo_batch(self, updates_per_combo, batch_size):
        losses = self.agent.learn_from_batch(self._traj_pool, updates_per_combo, batch_size)
        return losses
