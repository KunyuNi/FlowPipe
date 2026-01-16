import json
import math
import os

from loguru import logger

import comp
import deterministic
import env
import util
from ctxpipe.env.metric import *
from ctxpipe.env.primitives import *
from ctxpipe.info import Info

_info = None

VERSION = "gfnc_film"

def set_info(info: Info):
    global _info
    _info = info


def init():
    # TODO: Extract initializing configuration into object
    global _info
    if not _info:
        raise ValueError("info path not set")

    GlobalConfig.dataset_path = _info.dataset_prefix
    GlobalConfig.llm_embedding_path = _info.llm_embedding_prefix


    if os.path.exists(_info.aipipe_core_prefix):
        with open(_info.task_info_path) as f: # 'classification_task_dic.json'
            GlobalConfig.classification_task_dic = json.load(f)

        """load information files"""
        GlobalConfig.fold_length = math.ceil(
            len(GlobalConfig.classification_task_dic) / Config.k_fold
        ) #  计算每一折的长度

        with open(_info.task_index_path) as f: # test_index.json
            GlobalConfig.test_index = json.load(f)

        # GlobalConfig.train_index = list(
        #     set(GlobalConfig.classification_task_dic.keys()) #  完全相同�?
        #     - set(GlobalConfig.test_index) #  �?
        # ) # 计算训练集的索引  排除测试

        GlobalConfig.train_index=list(set(GlobalConfig.classification_task_dic.keys()))
        GlobalConfig.train_index.sort()

        # 计算“全集”大小：任务 × 预测器 × 逻辑管道
        combos_total = (
                len(GlobalConfig.train_index)
                * len(comp.predictors)
                * len(comp.lpipelines)  # ✅ 加上逻辑管道
        )

        if combos_total > 0:
            # 仅当未显式指定时，给出合理默认
            # ✅ 用 GFN 的 buffer_capacity 来估算每轮能承载的新轨迹数对应的组合上限
            #   （经验上限：每个组合采 traj_per_combo 条 → 可承载的组合 ≈ buffer_capacity / traj_per_combo）
            if getattr(default_gfn_config, "combos_per_epoch", 0) in (0, None):
                traj = max(1, default_gfn_config.traj_per_combo)
                gfn_limit = max(1, default_gfn_config.buffer_capacity // traj)
                # 给个保守目标：别超过全集，也别超过“缓冲可承载”的上限
                default_gfn_config.combos_per_epoch = min(combos_total, gfn_limit)
                min_buf = default_gfn_config.combos_per_epoch * default_gfn_config.traj_per_combo * 5
                if default_gfn_config.buffer_capacity < min_buf:
                    default_gfn_config.buffer_capacity = min_buf

            # 如果用户没给 num_epochs，就按“覆盖一遍”的轮数推导（与确定性覆盖调度对齐）
            if getattr(default_gfn_config, "num_epochs", 0) in (0, None):
                C = max(1, default_gfn_config.combos_per_epoch)
                default_gfn_config.num_epochs = math.ceil(combos_total / C)


class Config:
    k_fold = 3
    checkpoint: bool = True
    record: bool = False


class GlobalConfig:
    device = env.DEVICE
    enable_context_plugin = True
    enable_ocg_experience_replay = True

    ### hyperparameters for DQN
    gamma = 0.0
    learning_rate = 1e-5  # 1e-5
    frames = 50000
    max_buff = 5000

    column_num = 100

    step_timeout = 300 # 过期时间  测试时间可以时间长一点 训练300
    component_step_timeout = 200  # 单个组件执行的超时时间，可根据需要调整

    eps_decay = 2000

    blank_reward = 0.0 # 空白奖励
    blank_rewards_str = (
        f"{blank_reward}" if blank_reward >= 0.0 else f"m{-blank_reward}"
    )

    batch_size = 200 if not env.IS_TEST else 10
    logic_batch_size = batch_size // 5

    ctxpipe_setup_name = ""
    if not enable_context_plugin:
        ctxpipe_setup_name = "-noctx"
    elif not enable_ocg_experience_replay:
        ctxpipe_setup_name = "-noocg"
    else:
        ctxpipe_setup_name = "-3linear"

    # version = f"{'TEST_' if env.IS_TEST else ''}_{ctxpipe_setup_name}"
    """ 更改version"""
    version = f"{'TEST_' if env.IS_TEST else ''}{VERSION}"

    exp_dir = util.abspath(env.exp_prefix, f"{version}")
    log_dir = util.abspath("logs", f"{version}")
    model_dir: str = util.abspath("models", f"{version}")

    result_log_file_name: str = util.abspath(log_dir, "result_log.npy")
    loss_log_file_name: str = util.abspath(log_dir, "loss_log.pkl")
    lp_loss_log_file_name: str = util.abspath(log_dir, "lp_loss_log.npy")
    test_reward_dic_file_name: str = util.abspath(log_dir, "test_reward_dict.npy")

    pipelines_file_name: str = util.abspath(exp_dir, "diffprep.tsv")

    def makedirs(self):
        for d in [self.log_dir, self.exp_dir, self.model_dir]:
            logger.debug("making dir: {}", d)
            os.makedirs(d, exist_ok=True)

    backpropagate_interval: int = 50 if not env.IS_TEST else 10
    checkpoint_interval: int = 200

    column_feature_dim =  35   # 数据 特征
    data_dim: int = column_num * column_feature_dim

    classification_task_dic = {}

    dataset_path = "./data/dataset" # 默认路径 �?
    fold_length = -1 # ？？�?
    train_index = []
    test_index = []


default_config = GlobalConfig()


class AgentConfig(GlobalConfig):
    epsilon_start = 1.0 if not env.IS_TEST else 0.2
    epsilon_min = 0.2 if not env.IS_TEST else 0.1

    # prim_state_dim: int = GlobalConfig.data_dim + 6 + 1 + 1
    lpip_state_dim: int = GlobalConfig.data_dim + 1
    LLM_name="Meta-Llama-3.1-8B"



class DQNConfig(GlobalConfig):
    ### RNN param
    seq_embedding_dim: int = 96
    seq_hidden_size: int = 96
    seq_num_layers: int = 1
    predictor_embedding_dim: int = 16
    lpipeline_embedding_dim: int = 8





class GFlowNetConfig(GlobalConfig):
    # 定义 动作区间常量
    LOGIC_ACTION_START = 0
    LOGIC_ACTION_END = comp.num_lpipelines      # 不包含尾
    PRIM_ACTION_START = LOGIC_ACTION_END
    # 站位
    combos_per_epoch = 10               # 让 init() 来设置
    num_epochs = 500                    # 让 init() 来设置 ,默认全覆盖一遍就是 0

    use_gfn = True            # 1) 新增开
    learning_rate = 5e-4
    hidden_dim = 512
    trajectory_batch_size = 16
    buffer_capacity = 80000
    max_steps = 1 + max(len(lp) for lp in comp.lpipelines)
    frames = 30000
    log_interval = 20
    checkpoint_interval = 20
    # 块状训练节奏（推荐起点）
    coverage_seed = deterministic.RANDOM_SEED
    traj_per_combo = 8 # 每条轨迹采样多少次 -深度
    updates_per_combo = 1
    failure_weight = 0.2 #  失败轨迹 权重
    enable_reward_normalization = False  # 是否开启平滑
    reward_norm_momentum = 0.05       # EMA 平滑系数
    reward_temperature =2.0
    # 2) 统一动作空间：取所?Primitive gid 的最大�?+ 1
    inference_samples = 3  # 增加推理时的采样次数
    inference_temperature = 0.1  # 推理时的温度参数，更确定性
    # 增加上下文信息
    use_conditional_gfn: bool = True  #  条件 GFN
    use_attention: bool = True  # 仅当 use_conditional_gfn=True 时生效
    ctx_dim: int = 4096  # 与 LLM hidden_size 对齐
    LLM_name: str = "Meta-Llama-3.1-8B"  # 用于日志/加载
    use_learnable_ctx: bool = False  # 消融实验：使用可学习随机向量替代LLM embedding

    # 新增：推理模式和 Top-N
    inference_mode = "sample"        # "sample" / "greedy" / "topn"
    inference_top_n = 3

    MAX_PRIM_GID = max(
        [p.gid for p in comp.imputernums]
        +[p.gid for p in comp.imputercat]
        + [p.gid for p in comp.encoders]
        + [p.gid for p in comp.fpreprocessings]
        + [p.gid for p in comp.fengines]
        + [p.gid for p in comp.fselections]
    )
    action_dim = MAX_PRIM_GID + 1+comp.num_lpipelines

    # 3) 状态维�?= 数据特征 + 各类 one-hot
    state_dim = (
        GlobalConfig.data_dim
        + len(comp.logic_pipeline_1)
        + comp.num_predictors
        + action_dim
        + 1 # logic_selected_flag 逻辑管道是否被选择的标识符
        + comp.num_lpipelines  # logic choice one-hot
    )

    # 修改第221行：
    model_dir = util.abspath("models", f"{GlobalConfig.version}")
    device = env.DEVICE

default_gfn_config = GFlowNetConfig()
default_gfn_config.use_attention = default_gfn_config.use_attention and default_gfn_config.use_conditional_gfn


class EnvConfig(GlobalConfig):
    classification_metric_id: int = 1  # Default: F1; Current: Acc
    regression_metric_id: int = 4
    action_dim: int = GFlowNetConfig.action_dim
default_dqn_config = DQNConfig()
default_agent_config = AgentConfig()



CONFIG=default_agent_config # 测试
default_env_config = EnvConfig()
