import argparse
import gc
import json
import os
import time
import traceback
from glob import glob
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger
from typing import List, Optional, Tuple
import comp
import config
import env
import util
from config import default_config as conf
from config import default_env_config, default_gfn_config
from flowpipe.dataset import Dataset
from flowpipe.info import Info
from flowpipe.pipegen import PipelineGenerator
from flowpipe.stats import Stats, init_stats_db
from sklearn.metrics import precision_score, recall_score, f1_score
from flowpipe.env.pipeline import Pipeline
from copy import deepcopy


env.init()
COUNTER = 0


def get_primitive_by_name(name):
    # Search in all primitive lists in comp
    all_lists = [
        comp.imputernums,
        comp.imputercat,
        comp.encoders,
        comp.fpreprocessings,
        comp.fengines,
        comp.fselections,
        comp.predictors
    ]
    for lst in all_lists:
        for prim in lst:
            if getattr(prim, "name", "") == name:
                return prim
    return None


def calculate_more_metrics(dataset: Dataset, ai_sequence: List[str]):
    try:
        steps = []
        for name in ai_sequence:
            prim = get_primitive_by_name(name)
            if prim:
                steps.append(prim)
            else:
                logger.debug(f"[{dataset.name}] Could not find primitive for {name}")
        
        if not steps:
            logger.debug(f"[{dataset.name}] No valid steps found")
            return None, None, None

        pipeline = Pipeline(dataset.taskid, dataset.predictor, dataset.metric, conf, train=True)
        
        # 1. Load data
        pipeline.load_data(dataset.taskid)
        train_x, test_x, train_y, test_y = pipeline.train_x, pipeline.test_x, pipeline.train_y, pipeline.test_y
        logger.debug(f"[{dataset.name}] Data loaded: train={len(train_x)}, test={len(test_x)}")
        
        # 2. Transform loop
        for step in steps:
            try:
                train_x, test_x = step.transform(train_x, test_x, train_y)
                # Cleanup inf/nan like Pipeline does
                if isinstance(train_x, pd.DataFrame):
                    train_x = train_x.replace([float("inf"), float("-inf")], float("nan")).fillna(0)
                if isinstance(test_x, pd.DataFrame):
                    test_x = test_x.replace([float("inf"), float("-inf")], float("nan")).fillna(0)
            except Exception as e:
                logger.error(f"[{dataset.name}] Transform failed for {step.name}: {e}")
                return 0, 0, 0

        # 3. Predict
        try:
            pred_y = dataset.predictor.transform(train_x, train_y, test_x)
            logger.debug(f"[{dataset.name}] Prediction complete, {len(pred_y)} predictions")
            
            # 4. Calculate metrics
            if getattr(dataset.metric, "type", "Classifier") == "Classifier":
                 avg = 'weighted'
                 precision = precision_score(test_y, pred_y, average=avg, zero_division=0)
                 recall = recall_score(test_y, pred_y, average=avg, zero_division=0)
                 f1 = f1_score(test_y, pred_y, average=avg, zero_division=0)
                 logger.debug(f"[{dataset.name}] Metrics: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
                 return precision, recall, f1
            else:
                 logger.debug(f"[{dataset.name}] Not a classifier, skipping metrics")
                 return 0, 0, 0 
                 
        except Exception as e:
            logger.error(f"[{dataset.name}] Prediction/Metric failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0, 0, 0
            
    except Exception as e:
        logger.error(f"[{dataset.name}] calculate_more_metrics failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0, 0, 0


class CollectionBuilder:
    def __init__(self, info: Info) -> None:
        self._info = info

    def build(self):
        for dataset_dir in glob(os.path.join(self._info.dataset_prefix, "*", "")):
            self._parse_task_info(dataset_dir)

    def _parse_task_info(self, dataset_dir: str):
        global COUNTER
        COUNTER += 1
        dataset_name = os.path.basename(os.path.dirname(dataset_dir + "dummy"))
        ds_csv = os.path.join(dataset_dir, "data.csv")
        logger.info("processing dataset {} at {} (#{})", dataset_name, ds_csv, COUNTER)

        info = util.read_json(os.path.join(dataset_dir, "info.json"))
        label_name = info["label"]

        with open(ds_csv, "r", encoding="utf-8") as f:
            header = f.readline().strip()
        columns = [col.strip('"') for col in header.split(",")]
        try:
            label_index = columns.index(label_name)
        except ValueError:
            logger.warning("label column not found in dataset {}. skip", dataset_name)
            return

        dataset = Dataset(dataset_name, ds_csv, label_index)
        predictor_name = env.eval_predictor_name
        self._update_files(dataset, predictor_name)

    def _update_files(self, dataset: Dataset, predictor_name: str):
        dataset_info = util.read_json(self._info.dataset_info_path)
        ctasks = util.read_json(self._info.task_info_path)
        test_index = util.read_json(self._info.task_index_path)

        task_id = str(len(ctasks))
        task_name = (
            dataset.path.split("/")[-2]
            + "_"
            + predictor_name
            + "_"
            + str(dataset.label_column_id)
        )

        if not any(task.get("task_name") == task_name for task in ctasks.values()):
            ctasks[task_id] = dataset.make_task_info(predictor_name, task_name)

        if dataset.name not in dataset_info:
            dataset_info[dataset.name] = dataset.info

        if task_id not in test_index:
            test_index.append(task_id)

        util.write_json(self._info.dataset_info_path, dataset_info)
        util.write_json(self._info.task_info_path, ctasks)
        util.write_json(self._info.task_index_path, test_index)


class Setup:
    def __init__(self, info: Info) -> None:
        self._info = info
        init_stats_db(self._info.stats_db_file_path)

    def _init_info_path(self):
        config.set_info(self._info)
        config.init()

    def evaluate_pipeline(self, start: int, end: int = -1,interval: Optional[int] = None,dry_run: bool = False):
        self._init_info_path()
        self._reset_result_files()

        datasets = self._get_datasets()
        ctasks = self._get_ctasks()

        if end == -1:
            end = start

        if interval is None or interval <= 0:
            interval = default_gfn_config.checkpoint_interval

        epochs = list(range(start, end + 1, interval))
        if epochs and epochs[-1] != end:
            epochs.append(end)

        for epoch in epochs:
            logger.info("evaluating on epoch {}...", epoch)
            self._evaluate_epoch(epoch, datasets, ctasks, dry_run)
            logger.info("evaluating on epoch {} done", epoch)

    def _reset_result_files(self):
        with open(self._info.failed_file_path, "w", encoding="utf-8") as f:
            f.write(
                "iteration\tnotebook_path\tdataset_path\tlabel_index\tmodel\treason\ttraceback\n"
            )
        with open(self._info.done_file_path, "w", encoding="utf-8") as f:
            f.write("iteration\tnotebook_path\tdataset_path\tlabel_index\tmodel\tscore\n")

    def _get_datasets(self) -> dict:
        return util.read_json(self._info.dataset_info_path)

    def _get_ctasks(self) -> dict:
        with open(self._info.task_info_path, "r", encoding="utf-8") as f:
            classification_task_dic = json.load(f)
        mapping: dict[str, list[tuple[str, dict]]] = {}
        for task_id, info in classification_task_dic.items():
            mapping.setdefault(info["dataset"], []).append((task_id, info))
        return mapping

    def _evaluate_epoch(self, epoch: int, datasets: dict, ctasks: dict, dry_run: bool):
        # model_tag = f"epoch_{epoch}"
        model_tag = f"combo_{epoch}"
        logger.info("model_tag={}", model_tag)

        for dataset_name, info in datasets.items():
            tasks = ctasks.get(dataset_name)
            if not tasks:
                logger.warning("dataset {} has no task entries; skip", dataset_name)
                continue

            label_index = info.get("index")
            if isinstance(label_index, list):
                label_index = label_index[0] if label_index else 0

            task_entry = self._select_task_entry(tasks)
            if task_entry is None:
                logger.warning("dataset {} has no matched task; skip", dataset_name)
                continue

            task_id, task_info = task_entry
            csv_file = task_info["csv_file"]
            dataset_path = os.path.join(self._info.dataset_prefix, dataset_name, csv_file)

            predictor_obj = self._get_predictor_by_name(task_info["model"])
            metric_obj = self._get_metric()
            if predictor_obj is None or metric_obj is None:
                logger.warning("skip {} due to missing predictor/metric", dataset_name)
                continue

            dataset = Dataset(dataset_name, dataset_path, label_index)
            dataset.taskid = task_id
            dataset.predictor = predictor_obj
            dataset.metric = metric_obj

            if (
                not dry_run
                and Stats.select()
                .where((Stats.dataset == dataset_name) & (Stats.iteration == epoch))
                .exists()
            ):
                logger.warning("dataset {} already evaluated for epoch {}; skip", dataset_name, epoch)
                continue

            self._do_evaluate(dataset, task_info["model"], model_tag, dry_run)
            gc.collect()

    @staticmethod
    def _select_task_entry(tasks: List[Tuple[str, dict]]):
        preferred = [entry for entry in tasks if entry[1].get("model") == env.eval_predictor_name]
        return preferred[0] if preferred else (tasks[0] if tasks else None)

    @staticmethod
    def _get_predictor_by_name(predictor_name: str):
        for predictor in comp.predictors:
            name = getattr(predictor, "name", predictor.__class__.__name__)
            if name == predictor_name:
                return predictor
        return None

    @staticmethod
    def _get_metric():
        metric_id = default_env_config.classification_metric_id
        for metric in comp.metrics:
            if getattr(metric, "id", None) == metric_id:
                return metric
        return None

    def _do_evaluate(self, dataset: Dataset, model: str, model_tag: str, dry_run: bool):
        failed_f = done_f = None
        iteration = int(model_tag.split("_")[1])

        if not dry_run:
            failed_f = open(self._info.failed_file_path, "a", encoding="utf-8")
            done_f = open(self._info.done_file_path, "a", encoding="utf-8")

        try:
            pg = PipelineGenerator(dataset, model_tag)
            start = time.time()
            pg.generate()
            end = time.time()

            stats = pg.output()
            stats.execution_time = end - start
            stats.save()

            logger.info("dataset {} finished; score={:.6f}", dataset.name, pg.ml_score)
            # 新增：保存pipeline序列到pipelines.tsv
            if not dry_run:
                mode = getattr(default_gfn_config, "inference_mode", "sample")
                # 根据mode选择不同的文件名
                if mode == "greedy":
                    pipelines_file = conf.pipelines_file_name.replace("diffprep.tsv", "g_pipeline.tsv")
                else:
                    pipelines_file = conf.pipelines_file_name
                with open(pipelines_file, "a", encoding="utf-8") as f:
                    f.write(f"{model_tag}\t{dataset.name}\t{pg.ai_sequence}\t{pg.ml_score}")
                    
                    # Calculate extra metrics
                    prec, rec, f1 = calculate_more_metrics(dataset, pg.ai_sequence)
                    if prec is not None and (prec > 0 or rec > 0 or f1 > 0):
                        f.write(f"\t{prec:.6f}\t{rec:.6f}\t{f1:.6f}")
                    elif prec is not None:
                        # All zeros case - stil write but worth noting
                        f.write(f"\t{prec:.6f}\t{rec:.6f}\t{f1:.6f}")
                    
                    f.write("\n")

            if done_f:
                done_f.write(
                    f"{iteration}\t{dataset.name}\t"
                    f"{dataset.path}\t{dataset.label_column_id}\t{model}\t{pg.ml_score}\n"
                )

        except Exception as exc:
            tb = traceback.format_exc().replace("\n", "\\n")
            logger.error("evaluation failed for {}: {}", dataset.name, exc)
            if failed_f:
                failed_f.write(
                    f"{iteration}\t{dataset.name}\t"
                    f"{dataset.path}\t{dataset.label_column_id}\t{model}\t{exc}\t{tb}\n"
                )
        finally:
            if failed_f:
                failed_f.close()
            if done_f:
                done_f.close()


def parse_args():
    parser = argparse.ArgumentParser(description="GFN evaluation")
    parser.add_argument("--start_epoch", type=int, default=10)
    parser.add_argument("--end_epoch", type=int, default=2000)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def evaluate(info: Info, start: int, end: int, interval: int, dry_run: bool):
    default_gfn_config.use_gfn = True
    setup = Setup(info)
    CollectionBuilder(info).build()
    setup.evaluate_pipeline(start=start, end=end, interval=interval, dry_run=dry_run)
def evaluate_on_diffprep_dataset():
    # anoy
    args = parse_args()
    conf.pipelines_file_name = util.abspath(conf.exp_dir, "diffprep.tsv")
    # anoy
    evaluate(
        Info(
            aipipe_core_prefix=f"{conf.exp_dir}/aipipe", # conf.exp_dir flowpipe/exp/flowpipe-3linear/aipipe
            result_prefix=f"{conf.exp_dir}/result",
            dataset_prefix=f"data/diffprep_dataset",
            llm_embedding_prefix="LLM/llm_embedding/diffprep"
        ),
        start=int(args.start_epoch),  # 修正：start_step -> start_epoch
        end=int(args.end_epoch),  # 修正：end_step -> end_epoch
        interval=args.interval,
        dry_run=args.dry_run,
    )
def evaluate_on_deepline_dataset():
    # anoy
    # anoy
    args = parse_args()
    print(conf.exp_dir)
    conf.pipelines_file_name = util.abspath(conf.exp_dir, "deepline.tsv")
    # anoy
    evaluate(
        Info(
            aipipe_core_prefix=f"{conf.exp_dir}/deepline/aipipe", # conf.exp_dir flowpipe/exp/flowpipe-3linear/aipipe
            result_prefix=f"{conf.exp_dir}/deepline/result",
            dataset_prefix=f"data/deepline_dataset",
            llm_embedding_prefix="LLM/llm_embedding/deepline",
        ),
        start=int(0),  # 修正：start_step -> start_epoch
        end=int(30000),      # 修正：end_step -> end_epoch
        interval=args.interval,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    evaluate_on_diffprep_dataset()
    evaluate_on_deepline_dataset()
    print("counter=", COUNTER)
