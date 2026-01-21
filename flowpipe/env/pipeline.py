import multiprocessing
import json
import os
import signal
import time
from multiprocessing import Process
from typing import Optional, Tuple, List
from copy import deepcopy

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import comp
import util
from config import GlobalConfig,GFlowNetConfig
from .primitives.predictor import *
from .primitives.imputercat import ImputerCatPrim
from .primitives.primitive import Primitive


class FunctionTimedOut(Exception):
    pass

# anoy
def _do_add_step(
    train_x, test_x, train_y, step: Primitive, queue: multiprocessing.Queue
):
    os.setsid()
    print(f"executing function _do_add_step")

    try:
        train_x, test_x = step.transform(train_x, test_x, train_y)
        num_cols = list(train_x._get_numeric_data().columns) # 获取数值列
        cat_cols = list(set(train_x.columns) - set(num_cols)) # 获取类别�?
        queue.put([train_x, test_x, num_cols, cat_cols]) # 将处理后的数据放入队�?
    except Exception as e:
        print(f"ERROR in _do_add_step: {step.name} failed with: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        error_info = {
            'type': 'exception',
            'exception': e,
            'step_name': step.name,
            'error_msg': str(e)
        }
        queue.put(error_info)
    finally:
        queue.close()
    # return train_x, test_x, num_cols, cat_cols


def _do_evaluate(
        train_x, test_x, train_y, test_y, predictor, metric, queue: multiprocessing.Queue
):
    os.setsid()
    print(f"executing function _do_evaluate")

    try:
        pred_y = predictor.transform(train_x, train_y, test_x)
        result = metric.evaluate(pred_y, test_y)  # 评估模型 使用传入的mertric
        queue.put([pred_y, result])
    except Exception as e:
        # 关键：先打印异常信息到控制台，确保能看到
        # print(f"ERROR in _do_evaluate: Predictor {predictor.name} failed with: {str(e)}")
        # print(f"Exception type: {type(e).__name__}")

        error_info = {
            'type': 'exception',
            'exception': e,
            'predictor_name': predictor.name,
            'error_msg': str(e),
            'exception_type': type(e).__name__
        }
        queue.put(error_info)
        # 异常情况下不要执�?return，让函数自然结束
    finally:
        queue.close()
    # 移除这里�?return，避免干扰异常信息传�?


class Pipeline:

    def __init__(
        self, taskid, predictor: Primitive, metric, config: GlobalConfig, train=True
    ):
        self._config = config
        self.taskid = taskid
        self.metric = metric
        self.predictor = predictor
        self.train = train

        self.code = ""
        self.result = 0
        self.sequence: List[Primitive] = []
        self.index = 0

        self.data_x: pd.DataFrame = None
        self.data_y: pd.DataFrame = None
        self.train_x: pd.DataFrame = None
        self.test_x: pd.DataFrame = None
        self.train_y: pd.DataFrame = None
        self.test_y: pd.DataFrame = None
        self.pred_y: pd.DataFrame = None

        self.num_cols: list = []
        self.cat_cols: list = []

        self._last_eval_result = None
        self._eval_cache_valid = False

        self.load_data(taskid)
        self._logic_pipeline_id = None
        self.logic_pipeline_selected = False  # 标识符 是否已选择 逻辑管道
        self.gsequence = [0,0,0,0,0,0]

    @property
    def logic_pipeline_id(self) -> int:
        if self._logic_pipeline_id is None:
            raise ValueError("self.logic_pipeline_id not initialized")

        return self._logic_pipeline_id

    @logic_pipeline_id.setter
    def logic_pipeline_id(self, value) -> None:
        self._logic_pipeline_id = value

    def load_data(self, taskid, ratio=0.8, split_random_state=0):
        data = pd.read_csv(
            os.path.join(
                self._config.dataset_path,
                self._config.classification_task_dic[taskid]["dataset"],
                self._config.classification_task_dic[taskid]["csv_file"],
            )
        ).infer_objects()
        #  获取标签�?
        label_index = int(self._config.classification_task_dic[taskid]["label"])

        data = data.replace([np.inf, -np.inf], np.nan)
        data.dropna(subset=[data.iloc[:, label_index].name])
        # 如果数据量大�?500且是训练集，则只取前1500条数�?
        if data.shape[0] > 1500 and self.train:
            data = data.iloc[:1500, :]

        column = str(data.columns[label_index])
        # logger.debug(f"column={column}")
        self.data_x = data.drop(columns=[column], axis=1)
        self.data_y = data.iloc[:, label_index].values
        # if getattr(self.metric, "type", "").lower() == "classifier":
        #     le = LabelEncoder()
        #     self.data_y = le.fit_transform(self.data_y)
        del data
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.data_x,
            self.data_y,
            train_size=ratio,
            test_size=1 - ratio,
            random_state=split_random_state,
        )

        if str(self.data_y.dtype) == "Object":
            le = LabelEncoder()
            self.data_y = le.fit_transform(self.data_y)

        self.num_cols = list(self.train_x._get_numeric_data().columns)
        self.cat_cols = list(set(self.train_x) - set(self.num_cols))

    def reset_data(self):
        self.data_x = None
        self.data_y = None
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        self.pred_y = None

        del self.data_x
        del self.data_y
        del self.train_x
        del self.test_x
        del self.train_y
        del self.test_y
        del self.pred_y
        del self._config
        del self.taskid
        del self.metric
        del self.predictor
        del self.train
        del self.num_cols
        del self.cat_cols

        del self.code
        del self.result
        del self.sequence
        del self.index

    def get_index(self):
        return self.index

    def _subprocess(self, func, args, has_timeout, step_timeout=None):
        q = multiprocessing.Queue()
        args.append(q)
        process = Process(target=func, args=args)

        # logger.debug(f"process {func.__name__} created")

        func_return = None

        process.start()

        if has_timeout:
            timed_out = False
            finished = False
            passed_time = 0.0
            timeout_seconds = step_timeout or self._config.step_timeout
            # logger.debug(f"Current step_timeout: {timeout_seconds}")
            while not timed_out and not finished:
                if not process.is_alive():
                    finished = True
                    break

                if not q.empty():
                    finished = True
                    break

                time.sleep(0.1)
                passed_time += 0.1
                if passed_time % 10 < 0.001:
                    logger.debug(f"Passed {passed_time} secs")

                if passed_time > timeout_seconds:
                    timed_out = True

            if timed_out:
                logger.warning(f"Timed out: {func.__name__}")
                func_return = None
            else:
                try:
                    func_return = q.get_nowait()
                    if isinstance(func_return, BaseException):
                        raise func_return
                    # anoy
                    elif isinstance(func_return, dict) and func_return.get('type') == 'exception':
                        # 直接返回异常信息，让上层处理
                        return func_return
                except:
                    logger.warning(f"Error: {func.__name__}")
                    func_return = None
        else:
            func_return = q.get()

        try:
            if process.pid:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                logger.warning(f"Use process.terminate()")
                process.terminate()
        except:
            pass

        os.system(f"pkill -f '/home/{os.getlogin()}/anaconda3/envs/ctxpipe.*joblib'")

        q.close()

        util.clean_mem()

        return func_return

    def add_step(self, step: Primitive, has_timeout=True):  # step is a Primitive
        if self.index >= len(comp.lpipelines[self.logic_pipeline_id]):
            return -1
        total_steps = len(comp.lpipelines[self.logic_pipeline_id])

        pre_pipeline = []
        if self.index > 0:
            for ind in range(self.index):
                pre_pipeline.append(ind)
        if (
            step.type in pre_pipeline
            or not step.can_accept(self.train_x)
            or not step.can_accept(self.test_x)
            or (not step.is_needed(self.train_x) and not step.is_needed(self.test_x))
        ):
            return 0

        try:
            func_return = self._subprocess(
                _do_add_step,
                [self.train_x, self.test_x, self.train_y, step],
                has_timeout=has_timeout,
                step_timeout=getattr(
                    self._config, 'component_step_timeout', self._config.step_timeout
                ),
            )
            if func_return is None:
                logger.error(f"adding step {step} timed out")
                return -1
            # 关键修改：检查是否返回了异常信息  修改
            if isinstance(func_return, dict) and func_return.get('type') == 'exception':
                logger.error(f"Step {step.name} failed with error: {func_return['error_msg']}")
                logger.error(f"Exception details: {func_return['exception']}")
                return -1

            [self.train_x, self.test_x, self.num_cols, self.cat_cols] = func_return
        except FunctionTimedOut:
            logger.error(f"adding step {step} timed out")
            return -1
        #
        self.train_x = self.train_x.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.test_x = self.test_x.replace([np.inf, -np.inf], np.nan).fillna(0)
        #  先清理缓�?
        self._eval_cache_valid = False
        self.sequence.append(step)
        self.gsequence[self.index] = step.gid

        logger.debug(
            "[pipeline] add_step success step={} progress={}/{}",
            step.name,
            self.index + 1,
            total_steps,
        )

        self.index += 1
        return 1

    def reset_stage(self):
        """仅清理执行阶段状态，供逻辑选择后复位使用"""
        self.sequence = []
        self.index = 0
        self.gsequence = [0 for _ in self.gsequence]
        self._eval_cache_valid = False
        self._last_eval_result = None


    def evaluate(self, has_timeout=True):
        full_len = len(comp.lpipelines[self.logic_pipeline_id])
        if self._eval_cache_valid:
            return self._last_eval_result

        if len(self.sequence) < full_len:
            self.result=1e-6
            self._eval_cache_valid = False
            return None

        logger.info(
            f"evaluating {self.sequence} using predictor {self.predictor.name}..."
        )

        # --- Caching Logic Start ---
        cache_key = f"{self.taskid}_{self.predictor.name}_{self.gsequence}"
        cached_score = self._check_cache(cache_key)
        if cached_score is not None:
            self.result = cached_score
            self._last_eval_result = self.result
            self._eval_cache_valid = True
            logger.info(f"[pipeline] Cache Hit! task={self.taskid} key={cache_key} score={self.result}")
            return self.result
        # --- Caching Logic End ---

        try:
            func_return = self._subprocess(
                _do_evaluate,
                [
                    self.train_x,
                    self.test_x,
                    self.train_y,
                    self.test_y,
                    self.predictor,
                    self.metric,
                ],
                has_timeout=has_timeout,
            )
            # print(f"DEBUG: _subprocess returned: {type(func_return)} - {func_return}")
        except FunctionTimedOut:
            logger.error("evaluating %s timed out", self.sequence)
            self.result = 1e-6
            self._eval_cache_valid = False
            return self.result
        except Exception as e:
            logger.error("evaluation crashed: %s", e)
            self.result = 1e-6
            self._eval_cache_valid = False
            return self.result


        if func_return is None:
            logger.error("evaluation returned None")
            self.result = 1e-6
            self._eval_cache_valid = False
            return self.result

        if isinstance(func_return, dict) and func_return.get('type') == 'exception':
            error_msg = func_return.get("error_msg", "Unknown error")
            logger.error("evaluation failed:{}",error_msg)
            self.result = 1e-6
            self._eval_cache_valid = False
            return self.result

        [self.pred_y,metric] = func_return

        # 第378-384行修改为：
        try:
            reward = float(metric) if isinstance(metric, (int, float, str)) and str(metric).replace('.', '').replace(
                '-', '').isdigit() else 1e-6
        except (ValueError, TypeError):
            reward = 1e-6

        if isinstance(reward, (int, float)) and reward < 0:
            reward = 0
        elif not isinstance(reward, (int, float)):
            reward = 1e-6

        self.result = max(reward, 1e-6)

        # --- Update Cache ---
        self._update_cache(cache_key, self.result)
        # ------------------

        dataset_name = self._config.classification_task_dic[self.taskid]["dataset"]
        print("-------------当前使用",dataset_name,"---------------------")
        logger.info(
            "[pipeline] evaluate done task={} dataset={} predictor={} score={}",
            self.taskid,
            dataset_name,
            self.predictor.name,
            self.result,
        )
        self._last_eval_result = self.result
        self._eval_cache_valid = True
        return self.result

    # --- Cache Helper Methods ---
    _trajectory_cache = {}
    _cache_file_path = None

    @classmethod
    def _init_cache(cls, config):
        if cls._cache_file_path is None:
            cls._cache_file_path = os.path.join(config.log_dir, "trajectory_cache.json")
            if os.path.exists(cls._cache_file_path):
                try:
                    with open(cls._cache_file_path, 'r') as f:
                        cls._trajectory_cache = json.load(f)
                    logger.info(f"Loaded trajectory cache with {len(cls._trajectory_cache)} entries.")
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")
                    cls._trajectory_cache = {}

    def _check_cache(self, key):
        if self._cache_file_path is None:
            self._init_cache(self._config)
        return self._trajectory_cache.get(key)

    def _update_cache(self, key, score):
        if self._cache_file_path is None:
            self._init_cache(self._config)
        
        if key not in self._trajectory_cache:
            self._trajectory_cache[key] = score
            try:
                # Atomic write to prevent corruption
                temp_path = self._cache_file_path + ".tmp"
                with open(temp_path, 'w') as f:
                    json.dump(self._trajectory_cache, f)
                os.replace(temp_path, self._cache_file_path)
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

    # ------------------------------------------------------------------
    # GFlowNet helper interfaces
    # ------------------------------------------------------------------
    def _current_stage_name(self):
        if self._logic_pipeline_id is None:
            return None
        stages = comp.lpipelines[self._logic_pipeline_id]
        if self.index >= len(stages):
            return None
        return stages[self.index]

    def _stage_candidates(self, stage_name):
        if stage_name is None:
            return []
        if stage_name == "ImputerCat":
            return comp.imputercat
        mapping = {
            "ImputerNum": comp.imputernums,
            "Encoder": comp.encoders,
            "FeaturePreprocessing": comp.fpreprocessings,
            "FeatureEngine": comp.fengines,
            "FeatureSelection": comp.fselections,
        }
        return mapping.get(stage_name, [])

    def get_state_repr(self):
        data_feature = self.get_data_feature()
        if hasattr(data_feature, "tolist"):
            data_feature = data_feature.tolist()
        return {
            "logic_pipeline_selected": self.logic_pipeline_selected,
            "logic_pipeline_id": self._logic_pipeline_id
            if self._logic_pipeline_id is not None
            else -1,
            "stage_index": self.index if self.logic_pipeline_selected else -1,
            "selected_gids": [step.gid for step in self.sequence],
            "gsequence": list(self.gsequence),
            "predictor_id": self.predictor.id,
            "data_feature": data_feature,
        }


    """   
     def _is_primitive_valid(self, prim):
        can_accept_train = True
        can_accept_test = True
        needed_flag = True
        if self.train_x is not None:
            try:
                can_accept_train = prim.can_accept(self.train_x)
            except Exception:
                can_accept_train = False
            try:
                needed_flag = prim.is_needed(self.train_x)
            except Exception:
                needed_flag = True
        if self.test_x is not None:
            try:
                can_accept_test = prim.can_accept(self.test_x)
            except Exception:
                can_accept_test = False
            try:
                needed_flag = needed_flag or prim.is_needed(self.test_x)
            except Exception:
                pass
        return can_accept_train and can_accept_test and needed_flag"""
    def _is_primitive_valid(self, prim):
        stage_name = self._current_stage_name()

        # 统一识别 noop（各阶段�?noop gid 分别�?31~36�?
        is_noop = getattr(prim, "name", "") == "noop" or prim.gid in (31, 32, 33, 34, 35, 36)

        if (
            is_noop
            and stage_name is not None
            and self.train_x is not None
            and self.test_x is not None
        ):
            num_cols = list(self.train_x._get_numeric_data().columns)
            cat_cols = list(set(self.train_x.columns) - set(num_cols))

            num_cols_test_aligned = list(set(num_cols) & set(self.test_x.columns))
            cat_cols_test_aligned = list(set(cat_cols) & set(self.test_x.columns))

            with pd.option_context("mode.use_inf_as_na", True):
                num_has_nan = (
                        (len(num_cols) and self.train_x[num_cols].isna().any().any())
                        or (len(num_cols_test_aligned) and self.test_x[num_cols_test_aligned].isna().any().any())
                )
                cat_has_nan = (
                        (len(cat_cols) and self.train_x[cat_cols].isna().any().any())
                        or (len(cat_cols_test_aligned) and self.test_x[cat_cols_test_aligned].isna().any().any())
                )

            # 数值列仍有缺失 �?禁止 ImputerNum �?noop
            if stage_name == "ImputerNum" and num_has_nan:
                return False
            # 类别列仍有缺�?�?禁止 ImputerCat �?noop
            if stage_name == "ImputerCat" and cat_has_nan:
                return False
            # 仍存在非数值列 �?禁止 Encoder �?noop
            if stage_name == "Encoder" and len(cat_cols):
                return False

        can_accept_train = True
        can_accept_test = True
        needed_flag = True

        if self.train_x is not None:
            try:
                can_accept_train = prim.can_accept(self.train_x)
            except Exception:
                can_accept_train = False
            try:
                needed_flag = prim.is_needed(self.train_x)
            except Exception:
                needed_flag = True

        if self.test_x is not None:
            try:
                can_accept_test = prim.can_accept(self.test_x)
            except Exception:
                can_accept_test = False
            try:
                needed_flag = needed_flag or prim.is_needed(self.test_x)
            except Exception:
                pass

        return can_accept_train and can_accept_test and needed_flag

    def get_action_mask(self):
        if not self.logic_pipeline_selected:
            mask = [0] * self._config.action_dim
            for gid in range(comp.num_lpipelines):
                mask[gid] = 1
            return {
                "stage": "logic_selection",
                "mask": mask,
                "candidate_ids": list(range(comp.num_lpipelines)),
                "candidates": [],
            }

        stage_name = self._current_stage_name()
        candidates = self._stage_candidates(stage_name)
        mask = [0] * self._config.action_dim
        candidate_ids = []
        for prim in candidates:
            gid = prim.gid
            candidate_ids.append(gid)
            if 0 <= gid < len(mask) and self._is_primitive_valid(prim):
                mask[gid] = 1
        return {
            "stage": stage_name,
            "mask": mask,
            "candidate_ids": candidate_ids,
            "candidates": candidates,
        }

        # 逻辑已确定，继续原有 primitive 分支


    def apply_action(self, action_gid, has_timeout=True):
        if not self.logic_pipeline_selected:
            if not (
                GFlowNetConfig.LOGIC_ACTION_START <= action_gid < GFlowNetConfig.LOGIC_ACTION_END
            ):
                return self.get_state_repr(), True, -1
            self._logic_pipeline_id = int(action_gid)
            self.logic_pipeline_selected = True
            logger.debug("[pipeline] logic selected: {}", self._logic_pipeline_id)
            self.reset_stage()
            return self.get_state_repr(), False, 0

        stage_name = self._current_stage_name()
        candidates = self._stage_candidates(stage_name)
        dispatch = {prim.gid: prim for prim in candidates}

        if action_gid not in dispatch:
            return self.get_state_repr(), True, -1

        step = deepcopy(dispatch[action_gid])
        result = self.add_step(step, has_timeout=has_timeout)
        if result != 1:
            return self.get_state_repr(), True, result

        done = self.index >= len(comp.lpipelines[self.logic_pipeline_id])
        return self.get_state_repr(), done, 0


    def get_data_feature(self):
        """从Environment的get_data_feature方法复制过来，但适配Pipeline的上下文"""

        def _test_value(value) -> float:
            """处理 NaN 和 inf 值，返回 0.0"""
            if np.isnan(value) or abs(value) == np.inf:
                return 0.0
            else:
                return value

        def _test_frexp(value) -> Tuple[float, int]:
            """处理浮点值的尾数和指数"""
            if np.isnan(value) or abs(value) == np.inf:
                return 0.0, 0
            else:
                return np.frexp(value)

        inp_data = pd.DataFrame(self.train_x)

        column_info = {}

        for i in range(len(inp_data.columns)):
            col = inp_data.iloc[:, i]
            if i >= self._config.column_num:
                break
            s_s = col

            column_info[i] = {}
            column_info[i]["col_name"] = "unknown_" + str(i)
            column_info[i]["dtype"] = str(s_s.dtypes)  # 1
            column_info[i]["length"], column_info[i]["length_exp"] = _test_frexp(len(s_s.values))  # 2
            column_info[i]["null_ratio"] = s_s.isnull().sum() / len(s_s.values)  # 3
            column_info[i]["ctype"] = (
                1 if inp_data.columns[i] in self.num_cols else 2
            )  # 4
            column_info[i]["nunique"], column_info[i]["nunique_exp"] = _test_frexp(s_s.nunique())  # 5
            column_info[i]["nunique_ratio"] = s_s.nunique() / len(s_s.values)  # 6

            d = s_s.describe()

            if "mean" not in d:
                column_info[i]["ctype"] = 2  # 如果没有统计量，则认为是类别列

            # 数值型特征的统计量
            if column_info[i]["ctype"] == 1:  # numeric
                column_info[i]["mean"], column_info[i]["mean_exp"] = _test_frexp(d["mean"])  # 7
                column_info[i]["std"], column_info[i]["std_exp"] = _test_frexp(d["std"])  # 8
                column_info[i]["min"], column_info[i]["min_exp"] = _test_frexp(d["min"])  # 9
                column_info[i]["25%"], column_info[i]["25%_exp"] = _test_frexp(d["25%"])
                column_info[i]["50%"], column_info[i]["50%_exp"] = _test_frexp(d["50%"])
                column_info[i]["75%"], column_info[i]["75%_exp"] = _test_frexp(d["75%"])
                column_info[i]["max"], column_info[i]["max_exp"] = _test_frexp(d["max"])
                column_info[i]["median"], column_info[i]["median_exp"] = _test_frexp(s_s.median())

                if len(s_s.mode()) != 0:
                    column_info[i]["mode"], column_info[i]["mode_exp"] = _test_frexp(s_s.mode().iloc[0])
                else:
                    column_info[i]["mode"], column_info[i]["mode_exp"] = 0.0, 0

                # 计算模式比例，增加类别特征的频率信息
                mr = s_s.astype("category").describe().iloc[3] / len(s_s.values)
                column_info[i]["mode_ratio"] = _test_value(mr)

                column_info[i]["sum"], column_info[i]["sum_exp"] = _test_frexp(s_s.sum())
                column_info[i]["skew"], column_info[i]["skew_exp"] = _test_frexp(s_s.skew())
                column_info[i]["kurt"], column_info[i]["kurt_exp"] = _test_frexp(s_s.kurt())

            # 类别型特征的默认处理
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

        # 将所有特征整理成一个列表
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

            # 确保将新增的 mode_ratio 和 nunique_ratio 加入
            one_column_feature.append(column_dic.get("mode_ratio", 0.0))  # 追加 mode_ratio
            one_column_feature.append(column_dic.get("nunique_ratio", 0.0))  # 追加 nunique_ratio

            data_feature.append(one_column_feature)

        # 在数据不足时，填充零值
        if len(column_info) < self._config.column_num:
            for index in range(len(column_info), self._config.column_num):
                one_column_feature = np.zeros(self._config.column_feature_dim)
                data_feature.append(one_column_feature)

        # 统一数据格式为一维数组
        data_feature = np.ravel(np.array(data_feature))

        # 删除临时变量，释放内存
        del inp_data
        del column_info

        return data_feature

    # def get_data_feature(self):
    #     """从Environment的get_data_feature方法复制过来，但适配Pipeline的上下文"""
    #     def _test_value(value) -> float:
    #         if np.isnan(value) or abs(value) == np.inf:
    #             return 0.0
    #         else:
    #             return value
    #
    #     def _test_frexp(value) -> Tuple[float, int]:
    #         if np.isnan(value) or abs(value) == np.inf:
    #             return 0.0, 0
    #         else:
    #             return np.frexp(value)
    #
    #     inp_data = pd.DataFrame(self.train_x)
    #
    #     column_info = {}
    #
    #     for i in range(len(inp_data.columns)):
    #         col = inp_data.iloc[:, i]
    #         if i >= self._config.column_num:
    #             break
    #         s_s = col
    #
    #         column_info[i] = {}
    #         column_info[i]["col_name"] = "unknown_" + str(i)
    #         column_info[i]["dtype"] = str(s_s.dtypes)  # 1
    #         column_info[i]["length"], column_info[i]["length_exp"] = _test_frexp(
    #             len(s_s.values)
    #         )  # 2
    #         column_info[i]["null_ratio"] = s_s.isnull().sum() / len(s_s.values)  # 3
    #         column_info[i]["ctype"] = (
    #             1 if inp_data.columns[i] in self.num_cols else 2
    #         )
    #         column_info[i]["nunique"], column_info[i]["nunique_exp"] = _test_frexp(
    #             s_s.nunique()
    #         )  # 5
    #         column_info[i]["nunique_ratio"] = s_s.nunique() / len(s_s.values)  # 6
    #
    #         d = s_s.describe()
    #
    #         if "mean" not in d:
    #             column_info[i]["ctype"] = 2
    #
    #         if column_info[i]["ctype"] == 1:  # numeric
    #             column_info[i]["mean"], column_info[i]["mean_exp"] = _test_frexp(
    #                 d["mean"]
    #             )  # 7
    #             column_info[i]["std"], column_info[i]["std_exp"] = _test_frexp(
    #                 d["std"]
    #             )  # 8
    #             column_info[i]["min"], column_info[i]["min_exp"] = _test_frexp(
    #                 d["min"]
    #             )  # 9
    #             column_info[i]["25%"], column_info[i]["25%_exp"] = _test_frexp(d["25%"])
    #             column_info[i]["50%"], column_info[i]["50%_exp"] = _test_frexp(d["50%"])
    #             column_info[i]["75%"], column_info[i]["75%_exp"] = _test_frexp(d["75%"])
    #             column_info[i]["max"], column_info[i]["max_exp"] = _test_frexp(d["max"])
    #             column_info[i]["median"], column_info[i]["median_exp"] = _test_frexp(
    #                 s_s.median()
    #             )
    #
    #             if len(s_s.mode()) != 0:
    #                 column_info[i]["mode"], column_info[i]["mode_exp"] = _test_frexp(
    #                     s_s.mode().iloc[0]
    #                 )
    #             else:
    #                 column_info[i]["mode"], column_info[i]["mode_exp"] = 0.0, 0
    #
    #             mr = s_s.astype("category").describe().iloc[3] / len(s_s.values)
    #             column_info[i]["mode_ratio"] = _test_value(mr)
    #
    #             column_info[i]["sum"], column_info[i]["sum_exp"] = _test_frexp(
    #                 s_s.sum()
    #             )
    #             column_info[i]["skew"], column_info[i]["skew_exp"] = _test_frexp(
    #                 s_s.skew()
    #             )
    #             column_info[i]["kurt"], column_info[i]["kurt_exp"] = _test_frexp(
    #                 s_s.kurt()
    #             )
    #
    #         elif column_info[i]["ctype"] == 2:  # category
    #             column_info[i]["mean"], column_info[i]["mean_exp"] = 0.0, 0
    #             column_info[i]["std"], column_info[i]["std_exp"] = 0.0, 0
    #             column_info[i]["min"], column_info[i]["min_exp"] = 0.0, 0
    #             column_info[i]["25%"], column_info[i]["25%_exp"] = 0.0, 0
    #             column_info[i]["50%"], column_info[i]["50%_exp"] = 0.0, 0
    #             column_info[i]["75%"], column_info[i]["75%_exp"] = 0.0, 0
    #             column_info[i]["max"], column_info[i]["max_exp"] = 0.0, 0
    #             column_info[i]["median"], column_info[i]["median_exp"] = 0.0, 0
    #
    #             column_info[i]["mode"], column_info[i]["mode_exp"] = 0.0, 0
    #             column_info[i]["mode_ratio"] = 0.0
    #             column_info[i]["sum"], column_info[i]["sum_exp"] = 0.0, 0
    #             column_info[i]["skew"], column_info[i]["skew_exp"] = 0.0, 0
    #             column_info[i]["kurt"], column_info[i]["kurt_exp"] = 0.0, 0
    #
    #     data_feature = []
    #     for index in column_info.keys():
    #         one_column_feature = []
    #         column_dic = column_info[index]
    #         for kw in column_dic.keys():
    #             if kw == "col_name" or kw == "content":
    #                 continue
    #             elif kw == "dtype":
    #                 content = comp.dtype_id_map[column_dic[kw]]
    #             else:
    #                 content = column_dic[kw]
    #             one_column_feature.append(content)
    #         data_feature.append(one_column_feature)
    #
    #     if len(column_info) < self._config.column_num:
    #         for index in range(len(column_info), self._config.column_num):
    #             one_column_feature = np.zeros(self._config.column_feature_dim)
    #             data_feature.append(one_column_feature)
    #
    #     data_feature = np.ravel(np.array(data_feature))
    #
    #     del inp_data
    #     del column_info
    #
    #     return data_feature


