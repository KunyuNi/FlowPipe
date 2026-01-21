import os
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
import random
import comp
import deterministic
from config import EnvConfig

from .pipeline import Pipeline
from .primitives.primitive import Primitive


class Environment:

    def __init__(self, config: EnvConfig, train=True):
        self._config = config
        self.column_num = config.column_num
        self.state = None
        self.reward = None
        self.action = None
        self.next_state = None
        self.done = False
        self.train = train
        self.lpip_state = None
        self._pipeline = None

    @property
    def pipeline(self) -> Pipeline:
        if not self._pipeline:
            raise ValueError("self.pipeline not initialized")

        return self._pipeline
    # gfn reset
    def reset(self, taskid=None, predictor=None, metric=None,logic_pipeline_id= None  ,default=True):
        if self._pipeline is not None and taskid is None and predictor is None and metric is None and logic_pipeline_id is None:
            return

        print("taskid: {}, predictor: {}, metric: {}, default: {}".format(
            taskid, predictor, metric, default
        ))

        old_taskid = taskid
        old_predictor = predictor
        old_metric = metric

        is_pipeline_determined = False
        while not is_pipeline_determined:
            if default is True or taskid is None or predictor is None or metric is None:
                taskid = deterministic.task_rng.choice(self._config.train_index)
                logger.debug("taskid: {}".format(taskid))

                ids = [p.id for p in comp.predictors]
                predictor_id = deterministic.predictor_rng.choice(ids)
                try:
                    predictor = next(p for p in comp.predictors if p.id == predictor_id)
                except IndexError:
                    logger.warning("predictor_id {} not exist in comp.predictors. retrying".format(predictor_id))
                    taskid = old_taskid
                    continue

                logger.debug("predictor_id: {}".format(predictor_id))

                metric = [
                    i for i in comp.metrics
                    if i.id == self._config.classification_metric_id
                ][0]

            try:
                self.reset_pipeline(taskid, predictor, metric, train=self.train)
                #              # 逻辑选择改由 GFN 轨迹首个动作决定，这里不再预设 logic_pipeline_id
                # if logic_pipeline_id is not None:
                #     self._pipeline.logic_pipeline_id = logic_pipeline_id
                # elif getattr(self._pipeline, "_logic_pipeline_id", None) is None:
                #     self._pipeline.logic_pipeline_id = random.randrange(len(comp.lpipelines))
            except FileNotFoundError:
                dataset_path = os.path.join(
                    self._config.dataset_path,
                    self._config.classification_task_dic[taskid]["dataset"],
                    self._config.classification_task_dic[taskid]["csv_file"],
                )
                logger.warning("dataset {} not found. sample again".format(dataset_path))
                taskid = old_taskid
                predictor = old_predictor
                metric = old_metric
                continue

            is_pipeline_determined = True

            # >>> 关键：为 GFlowNet 初始化一个逻辑管线 ID <<<
            # if self._pipeline.logic_pipeline_id is None:
            #     self._pipeline.logic_pipeline_id = logic_pipeline_id
            # elif getattr(self._pipeline, "_logic_pipeline_id", None) is None:
            #     self._pipeline.logic_pipeline_id = random.randrange(len(comp.lpipelines))
            self.column_num = self._config.column_num
            self.lpip_state = self.get_lpip_state()
            self.reward = None
            self.action = None
            self.next_state = None
            self.done = False

    def reset_to_combo(self, taskid, predictor, metric, logic_pipeline_id=None):
        self.reset(
            taskid=taskid,
            predictor=predictor,
            metric=metric,
            logic_pipeline_id=None,
            default=False,
        )

    # DQN reset
    # def reset(self, taskid=None, predictor=None, metric=None, default=True):
    #     if self._pipeline is not None and taskid is None and predictor is None and metric is None:
    #         return
    #     print(
    #         f"taskid: {taskid}, predictor: {predictor}, metric: {metric}, default: {default}"
    #     )
    #
    #     old_taskid = taskid
    #     old_predictor = predictor
    #     old_metric = metric
    #     # DQN 和 GFN 逻辑不同
    #     is_pipeline_determined = False
    #     while not is_pipeline_determined:
    #         if default == True or taskid is None or predictor is None or metric is None:
    #             taskid = deterministic.task_rng.choice(self._config.train_index)  # json文件
    #             logger.debug(f"taskid: {taskid}")
    #             # 更改 anoy
    #             ids=[p.id for p in comp.predictors]
    #             predictor_id=deterministic.predictor_rng.choice(ids)
    #             try:
    #                 predictor = next(p for p in comp.predictors if p.id == predictor_id)
    #             except IndexError:
    #                 logger.warning(
    #                     f"predictor_id {predictor_id} not exist in comp.predictors. retrying"
    #                 )
    #                 taskid = old_taskid
    #                 continue
    #
    #             logger.debug(f"predictor_id: {predictor_id}")
    #
    #             metric = [
    #                 i
    #                 for i in comp.metrics
    #                 if i.id == self._config.classification_metric_id
    #             ][0]
    #
    #         try:
    #             self.reset_pipeline(taskid, predictor, metric, train=self.train)
    #         except FileNotFoundError:
    #             dataset_path = os.path.join(
    #                 self._config.dataset_path,
    #                 self._config.classification_task_dic[taskid]["dataset"],
    #                 self._config.classification_task_dic[taskid]["csv_file"],
    #             )
    #             logger.warning(f"dataset {dataset_path} not found. sample again")
    #             taskid = old_taskid
    #             predictor = old_predictor
    #             metric = old_metric
    #             continue
    #
    #         is_pipeline_determined = True
    #
    #         self.column_num = self._config.column_num
    #         self.lpip_state = self.get_lpip_state()
    #         self.reward = None
    #         self.action = None
    #         self.next_state = None
    #         self.done = False

    def reset_pipeline(self, taskid, predictor, metric, train=True):
        if self._pipeline:
            self._pipeline.reset_data()
            del self._pipeline
            self._pipeline = None

        self._pipeline = Pipeline(taskid, predictor, metric, self._config, train=train)

    def step(self, step: Primitive, has_timeout=True):

        logger.debug(f"adding step {step.name}...")

        if self._pipeline is None:
            self.reset()

        self.prim_state = self.get_state()

        add_result = self._pipeline.add_step(step,has_timeout=has_timeout)

        if add_result != 1:
            # 包含 timed out / can_accept 不通过 / 其它错误
            self.reward = 1e-6
            self.done = True
            self.action = step
            self.next_prim_state = self.get_state()
            return [self.prim_state, self.reward, self.next_prim_state, self.done], add_result

        self.next_prim_state = self.get_state()

        if len(self.pipeline.sequence) == len(comp.lpipelines[self.pipeline.logic_pipeline_id]):
            reward = self.pipeline.evaluate(has_timeout=has_timeout)
            self.reward = max(float(reward), 1e-6)
            self.done = True
        else:
            self.reward = 0.0  # 中间步骤不发奖励
            self.done = False

        self.action = step
        return [self.prim_state, self.reward, self.next_prim_state, self.done], None

    #
    def get_data_feature(self) -> np.ndarray:
        def _test_value(value) -> float:
            if np.isnan(value) or abs(value) == np.inf:
                return 0.0
            else:
                return value

        def _test_frexp(value) -> Tuple[float, int]:
            if np.isnan(value) or abs(value) == np.inf:
                return 0.0, 0
            else:
                return np.frexp(value)

        inp_data = pd.DataFrame(self.pipeline.train_x)

        # if len(self.pipeline.train_y.shape) > 1:
        #     train_y = self.pipeline.train_y[0]
        # else:
        #     train_y = self.pipeline.train_y
        # categorical = list(self.pipeline.train_x.dtypes == object)

        column_info = {}

        for i in range(len(inp_data.columns)):
            col = inp_data.iloc[:, i]
            if i >= self.column_num:
                break
            s_s = col

            column_info[i] = {}
            column_info[i]["col_name"] = "unknown_" + str(i)
            column_info[i]["dtype"] = str(s_s.dtypes)  # 1
            column_info[i]["length"], column_info[i]["length_exp"] = _test_frexp(
                len(s_s.values)
            )  # 2
            column_info[i]["null_ratio"] = s_s.isnull().sum() / len(s_s.values)  # 3
            column_info[i]["ctype"] = (
                1 if inp_data.columns[i] in self.pipeline.num_cols else 2
            )
            column_info[i]["nunique"], column_info[i]["nunique_exp"] = _test_frexp(
                s_s.nunique()
            )  # 5
            column_info[i]["nunique_ratio"] = s_s.nunique() / len(s_s.values)  # 6

            d = s_s.describe()

            if "mean" not in d:
                column_info[i]["ctype"] = 2

            if column_info[i]["ctype"] == 1:  # numeric
                column_info[i]["mean"], column_info[i]["mean_exp"] = _test_frexp(
                    d["mean"]
                )  # 7
                column_info[i]["std"], column_info[i]["std_exp"] = _test_frexp(
                    d["std"]
                )  # 8
                column_info[i]["min"], column_info[i]["min_exp"] = _test_frexp(
                    d["min"]
                )  # 9
                column_info[i]["25%"], column_info[i]["25%_exp"] = _test_frexp(d["25%"])
                column_info[i]["50%"], column_info[i]["50%_exp"] = _test_frexp(d["50%"])
                column_info[i]["75%"], column_info[i]["75%_exp"] = _test_frexp(d["75%"])
                column_info[i]["max"], column_info[i]["max_exp"] = _test_frexp(d["max"])
                column_info[i]["median"], column_info[i]["median_exp"] = _test_frexp(
                    s_s.median()
                )

                if len(s_s.mode()) != 0:
                    column_info[i]["mode"], column_info[i]["mode_exp"] = _test_frexp(
                        s_s.mode().iloc[0]
                    )
                else:
                    column_info[i]["mode"], column_info[i]["mode_exp"] = 0.0, 0

                mr = s_s.astype("category").describe().iloc[3] / len(s_s.values)
                column_info[i]["mode_ratio"] = _test_value(mr)

                column_info[i]["sum"], column_info[i]["sum_exp"] = _test_frexp(
                    s_s.sum()
                )
                column_info[i]["skew"], column_info[i]["skew_exp"] = _test_frexp(
                    s_s.skew()
                )
                column_info[i]["kurt"], column_info[i]["kurt_exp"] = _test_frexp(
                    s_s.kurt()
                )

                # print(f"column_info[{i}]: {column_info[i]}")

            elif column_info[i]["ctype"] == 2:  # category
                column_info[i]["mean"], column_info[i]["mean_exp"] = 0.0, 0
                column_info[i]["std"], column_info[i]["std_exp"] = 0.0, 0
                column_info[i]["min"], column_info[i]["min_exp"] = 0.0, 0
                column_info[i]["25%"], column_info[i]["25%_exp"] = 0.0, 0
                column_info[i]["50%"], column_info[i]["50%_exp"] = 0.0, 0
                column_info[i]["75%"], column_info[i]["75%_exp"] = 0.0, 0
                column_info[i]["max"], column_info[i]["max_exp"] = 0.0, 0
                column_info[i]["median"], column_info[i]["median_exp"] = 0.0, 0

                column_info[i]["mode"], column_info[i]["mode_exp"] = 0.0, 0
                column_info[i]["mode_ratio"] = 0.0
                column_info[i]["sum"], column_info[i]["sum_exp"] = 0.0, 0
                column_info[i]["skew"], column_info[i]["skew_exp"] = 0.0, 0
                column_info[i]["kurt"], column_info[i]["kurt_exp"] = 0.0, 0

        data_feature = []
        for index in column_info.keys():
            one_column_feature = []
            column_dic = column_info[index]
            for kw in column_dic.keys():
                if kw == "col_name" or kw == "content":
                    continue
                elif kw == "dtype":
                    content = comp.dtype_id_map[column_dic[kw]]
                else:
                    content = column_dic[kw]
                one_column_feature.append(content)
            data_feature.append(one_column_feature)

        if len(column_info) < self.column_num:
            for index in range(len(column_info), self.column_num):
                one_column_feature = np.zeros(self._config.column_feature_dim)
                data_feature.append(one_column_feature)

        data_feature = np.ravel(np.array(data_feature))

        del inp_data
        del column_info

        return data_feature

    def get_lpip_state(self) -> np.ndarray:
        data_feature = self.get_data_feature()
        predictor = np.array([self.pipeline.predictor.id])
        state = np.concatenate((data_feature, predictor))
        return state

    def get_state(self) -> np.ndarray:
        data_feature = self.get_data_feature()  #3300
        sequence = np.array(self.pipeline.gsequence) #6
        predictor = np.array([self.pipeline.predictor.id - 1]) #
        logic_pipeline_id = np.array([self.pipeline.logic_pipeline_id]) #1
        state = np.concatenate((data_feature, sequence, predictor, logic_pipeline_id))
        return state # 3308

    def gfn_get_state(self):
        if self._pipeline is None:
            self.reset(default=True)
        return self._pipeline.get_state_repr()

    def gfn_get_action_mask(self):
        if self._pipeline is None:
            self.reset(default=True)
        return self._pipeline.get_action_mask()

    def gfn_step(self, action_id, has_timeout=True):
        if self._pipeline is None:
            self.reset(default=True)
        next_state, done, err_code = self._pipeline.apply_action(
            action_id, has_timeout=has_timeout
        )
        if err_code != 0:
            return next_state, 0.0, True, err_code

        reward = 0.0
        if done:
            result = self._pipeline.evaluate(has_timeout=has_timeout)
            if result is None:
                result = self._pipeline.result
            reward = float(result)

        return next_state, reward, done, err_code

    def get_reward(self, has_timeout):
        if len(self.pipeline.sequence) < 6:
            self.reward = (
                0.0 if self.pipeline.sequence[-1].id == 0 else self._config.blank_reward
            )
        else:
            self.reward = self.pipeline.evaluate(has_timeout=has_timeout)
            self.end_time = time.time()

    def set_done(self):
        if len(self.pipeline.sequence) < 6:
            self.done = False
        elif len(self.pipeline.sequence) == 6:
            self.done = True
        else:
            return

    def has_nan(self):
        has_num_nan = False
        has_cat_nan = False

        def catch_num(data):
            num_cols = [
                col for col in data.columns if str(data[col].dtypes) != "object"
            ]
            cat_cols = [col for col in data.columns if col not in num_cols]
            cat_train_x = data[cat_cols]
            num_train_x = data[num_cols]
            return cat_train_x, num_train_x

        with pd.option_context("mode.use_inf_as_na", True):
            cat_train_x, num_train_x = catch_num(self.pipeline.train_x)
            cat_test_x, num_test_x = catch_num(self.pipeline.test_x)
            if len(self.pipeline.cat_cols) != 0:
                if cat_train_x.isna().any().any():
                    has_cat_nan = True
                if cat_test_x.isna().any().any():
                    has_cat_nan = True
            if len(self.pipeline.num_cols) != 0:
                if num_train_x.isna().any().any():
                    has_num_nan = True
                if num_test_x.isna().any().any():
                    has_num_nan = True

        return has_num_nan, has_cat_nan

    def has_cat_cols(self):
        if not len(self.pipeline.cat_cols) == 0:
            return True
        else:
            return False
