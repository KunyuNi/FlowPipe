import json
import math
import os

from loguru import logger

import comp
import deterministic
import env
import util
from flowpipe.env.metric import *
from flowpipe.env.primitives import *
from flowpipe.info import Info

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
        ) #  Calculate the length of each fold

        with open(_info.task_index_path) as f: # test_index.json
            GlobalConfig.test_index = json.load(f)

        # GlobalConfig.train_index = list(
        #     set(GlobalConfig.classification_task_dic.keys()) #  完全相同�?
        #     - set(GlobalConfig.test_index) #  �?
        # ) # 计算训练集的索引  排除测试

        GlobalConfig.train_index=list(set(GlobalConfig.classification_task_dic.keys()))
        GlobalConfig.train_index.sort()

        # Calculate "Total Set" size: Tasks x Predictors x Logic Pipelines
        combos_total = (
                len(GlobalConfig.train_index)
                * len(comp.predictors)
                * len(comp.lpipelines)  # ✅ Plus logic pipelines
        )

        if combos_total > 0:
            # Only give reasonable default if not explicitly specified
            # ✅ Use GFN's buffer_capacity to estimate the upper limit of combinations for new trajectories per round
            #   (Empirical limit: sample traj_per_combo per combo -> carriable combos ≈ buffer_capacity / traj_per_combo)
            if getattr(default_gfn_config, "combos_per_epoch", 0) in (0, None):
                traj = max(1, default_gfn_config.traj_per_combo)
                gfn_limit = max(1, default_gfn_config.buffer_capacity // traj)
                # Give a conservative goal: don't exceed full set, nor "buffer carriable" limit
                default_gfn_config.combos_per_epoch = min(combos_total, gfn_limit)
                min_buf = default_gfn_config.combos_per_epoch * default_gfn_config.traj_per_combo * 5
                if default_gfn_config.buffer_capacity < min_buf:
                    default_gfn_config.buffer_capacity = min_buf

            # If user didn't provide num_epochs, infer from "cover once" rounds (aligned with deterministic coverage scheduling)
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

    step_timeout = 300 # Expiration time. Testing time can be longer, training 300
    component_step_timeout = 200  # Detailed timeout for single component, adjustable

    eps_decay = 2000

    blank_reward = 0.0 # Blank reward
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
    """ Change version"""
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

    column_feature_dim =  35   # Data feature
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
    # Define action interval constants
    LOGIC_ACTION_START = 0
    LOGIC_ACTION_END = comp.num_lpipelines      # Exclusive of tail
    PRIM_ACTION_START = LOGIC_ACTION_END
    # Placeholder
    combos_per_epoch = 10               # Set by init()
    num_epochs = 500                    # Set by init(), default covering once is 0

    use_gfn = True            # 1) New: Enable
    learning_rate = 5e-4
    hidden_dim = 512
    trajectory_batch_size = 16
    buffer_capacity = 80000
    max_steps = 1 + max(len(lp) for lp in comp.lpipelines)
    frames = 30000
    log_interval = 20
    checkpoint_interval = 20
    # Block training rhythm (recommended start)
    coverage_seed = deterministic.RANDOM_SEED
    traj_per_combo = 8 # How many times to sample per trajectory - Depth
    updates_per_combo = 1
    failure_weight = 0.2 # Failed trajectory weight
    enable_reward_normalization = False  # Whether to enable smoothing
    reward_norm_momentum = 0.05       # EMA smoothing coefficient
    reward_temperature =2.0
    # 2) Unified action space: max of all Primitive gids + 1
    inference_samples = 3  # Increase samples during inference
    inference_temperature = 0.1  # Temperature parameter during inference, more deterministic
    # Add context information
    use_conditional_gfn: bool = True  # Conditional GFN
    use_attention: bool = True  # Only effective when use_conditional_gfn=True
    ctx_dim: int = 4096  # Align with LLM hidden_size
    LLM_name: str = "Meta-Llama-3.1-8B"  # For log/load
    use_learnable_ctx: bool = False  # Ablation experiment: use learnable random vector instead of LLM embedding

    # New: Inference mode and Top-N
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

    # 3) State dim = Data dim + Various one-hot
    state_dim = (
        GlobalConfig.data_dim
        + len(comp.logic_pipeline_1)
        + comp.num_predictors
        + action_dim
        + 1 # logic_selected_flag Identifier for whether logic pipeline is selected
        + comp.num_lpipelines  # logic choice one-hot
    )

    # Modify line 221:
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



CONFIG=default_agent_config # Test
default_env_config = EnvConfig()
