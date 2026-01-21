from enum import Enum, auto

import torch
from torch import nn

import comp
from config import DQNConfig
from config import default_gfn_config
from ctxpipe.env.primitives.primitive import Primitive
ENABLE_MODULO_HOOK = False
ENABLE_CONTEXT_GATE = True

N_DIM_EMBED = 4096 # 大模型嵌入维度
N_DIM_FIRST = 128
D = N_DIM_FIRST + N_DIM_EMBED
INFO_EXTRACTION_POS = 10
CTX_INTEGRATION_POS = 1

N_PRINT = 4


class ForwardMode(Enum):
    CLOSED = auto()
    GATED = auto()
    OPEN = auto()


def make_ctx_plugin_model(n_input: int, n_output: int) -> nn.Module:
    #  ctx_embedding 4096 -> 128
    result = nn.Sequential(
        nn.Linear(n_input, N_DIM_FIRST),
        nn.LeakyReLU(),
        # nn.Linear(N_DIM_FIRST, N_DIM_FIRST),
        # nn.LeakyReLU(),
        # nn.Linear(128, 64),
        # nn.LeakyReLU(),
        # nn.Linear(64, 32),
        # nn.LeakyReLU(),
        nn.Linear(N_DIM_FIRST, n_output),
        nn.Tanh(),
    )

    return result

    # encoder_layer = nn.TransformerEncoderLayer(d_model=N_DIM_EMBED, nhead=8, dim_feedforward=512, batch_first=True)
    # return nn.TransformerEncoder(encoder_layer, num_layers=4)


def make_mm_layer(shape: tuple, device: torch.device) -> torch.Tensor:
    MEAN = 3 / shape[0]
    result = torch.normal(mean=MEAN, std=MEAN / 3, size=shape).to(device)

    result.requires_grad_(True)
    return result


def forward_context_gate(model, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
    # print(x.shape, ctx.shape, model.context_gate.shape)
    # 门控因子的 计算函数
    info = torch.concat([x, ctx], dim=1)

    info = torch.matmul(info, model.context_gate)
    print("C MTML:", info.shape, info[:N_PRINT, :N_PRINT])

    info = info + model.context_gate_bias
    info = torch.nan_to_num(info)
    print("C +BAS:", info.shape, info[:N_PRINT, :N_PRINT])

    info = torch.sigmoid(info)
    print("C GATE:", info.shape, info[:N_PRINT, :N_PRINT])
    print("----------------------------------------")

    return info


class DQN(nn.Module):

    def __init__(self, num_inputs, actions_dim, config: DQNConfig):
        super(DQN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, N_DIM_FIRST),
            nn.LeakyReLU(),
            nn.Linear(N_DIM_FIRST, N_DIM_FIRST),
            nn.LeakyReLU(),
            nn.Linear(N_DIM_FIRST, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, actions_dim),
            nn.Tanh(),
        )

        self._config = config
        self.enable_ctxpipe = config.enable_context_plugin

        if self.enable_ctxpipe:
            self.ctx_linear = make_ctx_plugin_model(
                n_input=N_DIM_EMBED, n_output=actions_dim
            )
            # self.context_gate_model = make_ctx_gate_model(n_input=actions_dim + N_DIM_FIRST)
            self.context_gate = make_mm_layer(
                (actions_dim + N_DIM_FIRST, actions_dim), device=config.device
            )
            self.context_gate_bias = make_mm_layer((actions_dim,), device=config.device)

    def forward(
        self, x: torch.Tensor, ctx: torch.Tensor, mode: ForwardMode = ForwardMode.GATED
    ):
        print("DQN Forward")

        device = self._config.device

        data_feature = x[:, : self._config.data_dim].to(device)
        print("data", data_feature.shape, data_feature) #[1,1,3301]

        x = torch.concat([data_feature, x[:, self._config.data_dim :]], dim=-1)
        input_feature = x

        if not self.enable_ctxpipe:
            input_feature = self.nn(input_feature)
            print(f"No context: {input_feature[0]}")
            return input_feature

        ### CTXPIPE
        for i in range(len(self.nn)):
            input_feature = self.nn[i](input_feature)

            if i == len(self.nn) - INFO_EXTRACTION_POS:
                if ENABLE_CONTEXT_GATE:
                    ctx_integration = self.ctx_linear(ctx)

                    if mode == ForwardMode.CLOSED:
                        ctx_gate = torch.zeros(ctx_integration.shape).to(device)
                        ctx_gate.requires_grad_(False)

                    elif mode == ForwardMode.GATED:
                        ctx_gate = forward_context_gate(
                            self, input_feature, ctx_integration
                        ) #

                    elif mode == ForwardMode.OPEN:
                        ctx_gate = torch.ones(ctx_integration.shape).to(device)
                        ctx_gate.requires_grad_(False)

                    else:
                        raise ValueError(f"No such mode: {mode}")

                    ctx_integration = torch.mul(ctx_integration, ctx_gate)

            if i == len(self.nn) - CTX_INTEGRATION_POS:
                if ENABLE_CONTEXT_GATE:
                    print(f"Before gate: {input_feature[0]}")
                    print(f"Context: {ctx_integration[0]}")
                    print(f"Gate: {ctx_gate[0]}")
                    ctx_integration = torch.mul(ctx_integration - 1.0, ctx_gate) + 1.0
                    print(f"Context after gate: {ctx_integration[0]}")
                    input_feature = input_feature * ctx_integration
                    print(f"With context: {input_feature[0]}")

                if ENABLE_MODULO_HOOK:
                    input_feature.register_hook(self.modulo_func_hook)

        return input_feature


class RnnDQN(nn.Module):

    def __init__(self, actions_dim, config: DQNConfig):
        super(RnnDQN, self).__init__()

        self._config = config
        self.enable_ctxpipe = config.enable_context_plugin

        # input dim
        self.data_feature_dim = self._config.data_dim
        self.seq_feature_dim = len(comp.logic_pipeline_1)

        # seq_embedding_param
        # prim_nums = (
        #     len(
        #         set(comp.imputernums)
        #         | set(comp.encoders)
        #         | set(comp.fpreprocessings)
        #         | set(comp.fengines)
        #         | set(comp.fselections)
        #     )
        #     + 1
        #     + 1
        # )
        prim_gids=set()
        for arr in (comp.imputernums, comp.encoders, comp.fpreprocessings, comp.fengines, comp.fselections):
            for prim in arr:
                prim_gids.add(prim.gid)
                # print(f"DEBUG: {prim.__class__.__name__} gid={prim.gid}")

        #
        # prim_gids.add(Primitive().gid)
        # print(f"DEBUG: Blank Primitive gid={Primitive().gid}")
        # print(f"DEBUG: All prim_gids: {sorted(prim_gids)}")
        prim_nums = max(prim_gids) + 1
        # print(f"DEBUG: prim_nums (seq_embedding vocab size): {prim_nums}")


        seq_embedding_dim = config.seq_embedding_dim
        # seq_lstm param
        seq_hidden_size = config.seq_hidden_size
        seq_num_layers = config.seq_num_layers
        # predictor param
        predictor_embedding_dim = config.predictor_embedding_dim
        # lpip param
        self.lpipeline_nums = comp.num_lpipelines
        lpipeline_embedding_dim = config.lpipeline_embedding_dim

        # sequence networks
        self.seq_embedding = nn.Embedding(prim_nums, seq_embedding_dim,padding_idx=0)

        self.seq_lstm = nn.LSTM(
            input_size=seq_embedding_dim,
            hidden_size=seq_hidden_size,
            num_layers=seq_num_layers,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )

        # predictor
        self.predictor_embedding = nn.Embedding(
            comp.num_predictors, predictor_embedding_dim
        )

        # logic pipeline
        self.lpipeline_embedding = nn.Embedding(
            comp.num_lpipelines, lpipeline_embedding_dim
        )

        self.nn = nn.Sequential(
            nn.Linear(
                self.data_feature_dim
                + seq_hidden_size * self.seq_feature_dim
                + predictor_embedding_dim
                + lpipeline_embedding_dim,
                N_DIM_FIRST,
            ),
            nn.LeakyReLU(),
            nn.Linear(N_DIM_FIRST, N_DIM_FIRST),
            nn.LeakyReLU(),
            nn.Linear(N_DIM_FIRST, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, actions_dim),
            nn.Tanh(),
        )

        if self.enable_ctxpipe:
            self.ctx_linear = make_ctx_plugin_model(
                n_input=N_DIM_EMBED, n_output=actions_dim
            )
            self.context_gate = make_mm_layer(
                (actions_dim + N_DIM_FIRST, actions_dim), device=config.device
            )
            self.context_gate_bias = make_mm_layer((actions_dim,), device=config.device)

    def forward(
        self, x, ctx: torch.Tensor, mode: ForwardMode = ForwardMode.GATED
    ):  # x batch_size * state
        print("RnnDQN Forward")

        device = self._config.device

        data_feature = x[:, : self.data_feature_dim].to(
            device
        )  # (batch_size , data_dim))

        seq_feature = (
            x[:, self.data_feature_dim : self.data_feature_dim + self.seq_feature_dim]
            .type(torch.LongTensor)
            .to(device)
        )  # (batch_size , 6)
    # DEBUG
    #
    #     # 检查是否有超出范围的值
    #     if seq_feature.max() >= self.seq_embedding.num_embeddings:
    #         print(f"ERROR: seq_feature contains out-of-bounds values!")
    #         print(f"Max value: {seq_feature.max()}, vocab size: {self.seq_embedding.num_embeddings}")
    #     seq_feature = seq_feature.clamp_(min=0, max=self.seq_embedding.num_embeddings - 1)
    #DEBUG
        predictor_feature = (
            x[
                :,
                self.data_feature_dim
                + self.seq_feature_dim : self.data_feature_dim
                + self.seq_feature_dim
                + 1,
            ]
            .type(torch.LongTensor)
            .to(device)
        )  # (batch_size )

        lpipeline_feature = (
            x[
                :,
                self.data_feature_dim
                + self.seq_feature_dim
                + 1 : self.data_feature_dim
                + self.seq_feature_dim
                + 2,
            ]
            .type(torch.LongTensor)
            .to(device)
        )  # (batch_size )

        seq_embed_feature = self.seq_embedding(
            seq_feature
        )  # (batch_size , 6 , seq_embedding_dim)

        seq_hidden_feature, (h_1, c_1) = self.seq_lstm(
            seq_embed_feature
        )  # (6 , batch_size , seq_hidden_size)

        seq_hidden_feature = torch.flatten(
            seq_hidden_feature, start_dim=1
        )  # (batch_size , 6 * seq_hidden_size)

        predictor_embed_feature = self.predictor_embedding(
            predictor_feature
        )  # (batch_size, 1, predictor_embedding_dim)

        lpipeline_embed_deature = self.lpipeline_embedding(
            lpipeline_feature
        )  # (batch_size, 1, predictor_embedding_dim)

        predictor_embed_feature = torch.flatten(predictor_embed_feature, start_dim=1)
        lpipeline_embed_deature = torch.flatten(lpipeline_embed_deature, start_dim=1)

        input_feature = torch.cat(
            (
                data_feature,
                seq_hidden_feature,
                predictor_embed_feature,
                lpipeline_embed_deature,
            ),
            1,
        )

        if not self.enable_ctxpipe:
            input_feature = self.nn(input_feature)
            print(f"No context: {input_feature[0]}")
            return input_feature

        ### CTXPIPE
        for i in range(len(self.nn)):
            input_feature = self.nn[i](input_feature)

            ### S3
            if i == len(self.nn) - INFO_EXTRACTION_POS:
                if ENABLE_CONTEXT_GATE:
                    ctx_integration = self.ctx_linear(ctx)

                    if mode == ForwardMode.CLOSED:
                        ctx_gate = torch.zeros(ctx_integration.shape).to(device)
                        ctx_gate.requires_grad_(False)

                    elif mode == ForwardMode.GATED:
                        ctx_gate = forward_context_gate(
                            self, input_feature, ctx_integration
                        )

                    elif mode == ForwardMode.OPEN:
                        ctx_gate = torch.ones(ctx_integration.shape).to(device)
                        ctx_gate.requires_grad_(False)

                    else:
                        raise ValueError(f"No such mode: {mode}")

                    ctx_integration = torch.mul(ctx_integration, ctx_gate)

            if i == len(self.nn) - CTX_INTEGRATION_POS:
                if ENABLE_CONTEXT_GATE:
                    print(f"Before gate: {input_feature[0]}")
                    print(f"Context: {ctx_integration[0]}")
                    print(f"Gate: {ctx_gate[0]}")
                    ctx_integration = torch.mul(ctx_integration - 1.0, ctx_gate) + 1.0
                    print(f"Context after gate: {ctx_integration[0]}")
                    input_feature = input_feature * ctx_integration
                    print(f"With context: {input_feature[0]}")

                if ENABLE_MODULO_HOOK:
                    input_feature.register_hook(self.modulo_func_hook)

        return input_feature


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, ctx_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        if default_gfn_config.use_learnable_ctx:
            self.ctx_proj = nn.Sequential(
                nn.Linear(ctx_dim, hidden_dim),  # 降维/特征提取
                nn.ReLU(),  # 引入非线性：这一步很关键！
                nn.Linear(hidden_dim, hidden_dim * 2)  # 生成最终参数
            )
            with torch.no_grad():
                # ctx_proj是Sequential, 最后一层(索引2)是Linear
                self.ctx_proj[2].weight.fill_(0)
                self.ctx_proj[2].bias.fill_(0)
                # 让 gamma 初始为 1 (bias 的前一半设为 1)
                self.ctx_proj[2].bias[:hidden_dim].fill_(1.0)
        else:
            self.ctx_proj = nn.Linear(ctx_dim, hidden_dim * 2)
            with torch.no_grad():
                self.ctx_proj.weight.fill_(0)
                self.ctx_proj.bias.fill_(0)
                # 让 gamma 初始为 1 (bias 的前一半设为 1)
                self.ctx_proj.bias[:hidden_dim].fill_(1.0)

    def forward(self, x, ctx):
        x = self.layer(x)

        # 计算 FiLM 参数
        # ctx: (B, ctx_dim) -> (B, 2 * hidden_dim)
        params = self.ctx_proj(ctx)
        gamma, beta = torch.chunk(params, 2, dim=-1)

        # 应用调制: x = gamma * x + beta
        return x * gamma + beta

class FlowNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128,
                 ctx_dim=None, use_conditional=None, use_attention=None):
        super().__init__()
        self.use_conditional = default_gfn_config.use_conditional_gfn if use_conditional is None else use_conditional
        cdim = ctx_dim or default_gfn_config.ctx_dim

        # 1. 基础特征提取 (State -> Hidden)
        self.state_encoder = nn.Linear(state_dim, hidden_dim)

        # 2. FiLM 融合层 (替代原来的 backbone)
        # 如果启用条件，使用 FiLM；否则使用普通 Linear
        if self.use_conditional:
            self.film1 = FiLMLayer(hidden_dim, hidden_dim, cdim)
            self.film2 = FiLMLayer(hidden_dim, hidden_dim, cdim)
        else:
            self.layer1 = nn.Linear(hidden_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, hidden_dim)

        self.activation = nn.LeakyReLU()

        # 3. 动作头
        self.action_head = nn.Linear(hidden_dim, action_dim)

        # 4. Log Z 头 (修复：输入 cdim)
        if self.use_conditional:
            self.logz_mlp = nn.Sequential(
                nn.Linear(cdim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.log_z = nn.Parameter(torch.tensor(0.0))

    def forward(self, state, ctx=None):
        # 1. 编码状态
        x = self.activation(self.state_encoder(state))

        # 2. 应用中间层 (FiLM 或 普通层)
        if self.use_conditional and ctx is not None:
            x = self.activation(self.film1(x, ctx))
            x = self.activation(self.film2(x, ctx))
        else:
            # 兼容无条件模式
            layer1 = getattr(self, 'layer1', self.film1.layer if hasattr(self, 'film1') else None)
            layer2 = getattr(self, 'layer2', self.film2.layer if hasattr(self, 'film2') else None)
            x = self.activation(layer1(x))
            x = self.activation(layer2(x))

        # 3. 输出 Logits
        logits = self.action_head(x)

        # 4. 输出 Log Z (修复：不要 mean!)
        if self.use_conditional and ctx is not None:
            # 输出 (B, ) 而不是标量
            log_z = self.logz_mlp(ctx).squeeze(-1)
        else:
            # 广播到 Batch 维度
            log_z = self.log_z.expand(state.size(0))

        return logits, log_z




