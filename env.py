import os
import warnings
from typing import Literal

import numpy as np
import torch

import deterministic
from flowpipe.env.primitives import *


def init():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    deterministic.seed_everything()

    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)

    warnings.filterwarnings("ignore")

    from config import default_config as conf

    conf.makedirs()


IS_TEST = "TEST" in os.environ and os.environ["TEST"] != 0
DEVICE = torch.device("cuda:0")
#  支持运转的模型
supported_model = [
    "RandomForestClassifier",
    "KNeighborsClassifier",
    "LogisticRegression",
    "SVC",
    "MLPClassifier",
]


eval_predictor_id = 2 # 预定义 KNN ？
eval_predictor_name = supported_model[eval_predictor_id]

exp_prefix: str = f"exp"
