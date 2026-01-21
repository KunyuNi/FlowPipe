#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练 GFlowNet 的入口脚本（兼容 Python 3.8）
"""
import logging
import os
import sys
import time

import env

env.init()

import config as conf  # noqa: E402
from config import default_gfn_config  # noqa: E402
from flowpipe.agentman import AgentManager  # noqa: E402
from flowpipe.info import Info  # noqa: E402
from flowpipe.stats import init_stats_db  # noqa: E402

logging.basicConfig(level=logging.INFO)


class GFNSetup(object):
    """与 train.py 中的 Setup 保持一致，只是切换成 GFlowNet 配置"""

    def __init__(self, aipipe_core_prefix, result_prefix, dataset_prefix,llm_embedding_prefix ):
        self._info = Info(aipipe_core_prefix, result_prefix, dataset_prefix, llm_embedding_prefix)
        init_stats_db(self._info.stats_db_file_path)

    def _init_info_path(self):
        conf.set_info(self._info)
        conf.init()

    def train(self, resume_from=0, frames=None):
        self._init_info_path()

        if frames is not None:
            default_gfn_config.frames = frames

        default_gfn_config.use_gfn = True

        am = AgentManager()
        am.train(resume_from=resume_from)


def _parse_cli_args():
    resume_from = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
    frames = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    return resume_from, frames


def train_on_haipipe_dataset():
    resume_from, frames = _parse_cli_args()
    setup = GFNSetup(
        aipipe_core_prefix="data/meta",
        result_prefix="data/train_result",
        dataset_prefix="data/dataset",
        llm_embedding_prefix="LLM/llm_embedding/dataset",
    )
    logging.info("RESUME FROM %s", resume_from)
    if frames is not None:
        logging.info("FRAMES set to %s", frames)
    setup.train(resume_from=resume_from, frames=frames)


def train_on_diffprep_dataset():
    resume_from, frames = _parse_cli_args()
    setup = GFNSetup(
        aipipe_core_prefix="data/meta_diff",
        result_prefix="data/train_result",
        dataset_prefix="data/diffprep_dataset",
        llm_embedding_prefix="LLM/llm_embedding/diffprep"
    )
    logging.info("RESUME FROM %s", resume_from)
    if frames is not None:
        logging.info("FRAMES set to %s", frames)
    setup.train(resume_from=resume_from, frames=frames)


if __name__ == "__main__":
    start_time = time.time()
    train_on_haipipe_dataset()

    elapsed_time = time.time() - start_time
    logging.info("训练耗时: %.2f 小时", elapsed_time / 3600.0)
