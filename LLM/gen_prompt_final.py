#!/usr/bin/env python
# -*- coding: utf-8 -*-



import os
import json
import textwrap
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy import stats
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from typing import Optional
import argparse

warnings.filterwarnings('ignore')


# ============================ Global Config ============================

# 默认值，实际值会在 main 中根据 args 修改
DATA_ROOT = "/data1/data_1/anoy/anoy_pipe/data/dataset"
DATA_ROOT = "/data1/data_1/anoy/anoy_pipe/data/deepline_dataset"

OUTPUT_ROOT = "/data1/data_1/anoy/anoy_pipe/LLM/prompt_outputs"
# CONFIG_PATH = "/data1/data_1/anoy/anoy_pipe/data/meta/classification_task_dic.json"
CONFIG_PATH = "/data1/data_1/anoy/anoy_pipe/data/diffprep_dataset/classification_task_dic.json"

# CONFIG_PATH = "/data1/data_1/anoy/anoy_pipe/data/meta/classification_task_dic.json"

REP_SAMPLES = 100
REP_CLUSTERS = 8
RANDOM_STATE = 42

PROMPT_STYLES = [
    "contextual_semantic"
    # "statistical_sampling",
    # "raw_data",
    # "pipeline_component",
]

MISSING_THRESHOLD = 0.5
MAX_CORR_ROWS = 50_000
MINIBATCH_SWITCH_ROWS = 50_000
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

SENTINEL_TOKEN = "[CTX_END]"
TARGET_LAYER_INDEX = -1 

# =============================== Utils ================================
# ============================== Main =================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Prompts for CtxPipe")
    parser.add_argument(
        "--dataset",
        type=str,
        default="deepline",
        choices=["haipipe", "diffprep","deepline"],
        help="指定数据集: haipipe (训练集) 或 diffprep (测试集)"
    )
    return parser.parse_args()
def _load_dataset_config(config_path: str) -> dict:
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 无法加载配置文件 {config_path}: {e}")
        return {}

def _assess_data_quality(df: pd.DataFrame) -> dict:
    n_rows = len(df)
    n_cols = len(df.columns)
    isnull_sum = df.isnull().sum()
    nunique = df.nunique(dropna=True)

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include="object").columns

    high_missing_cols, constant_cols, high_card_cols = [], [], []
    for col in df.columns:
        missing_pct = (isnull_sum[col] / n_rows) * 100 if n_rows > 0 else 0.0
        if missing_pct > MISSING_THRESHOLD * 100:
            high_missing_cols.append((col, missing_pct))
        if nunique[col] == 1:
            constant_cols.append(col)
        elif df[col].dtype == "object" and nunique[col] > n_rows * 0.8:
            high_card_cols.append((col, int(nunique[col])))

    return {
        "total_rows": n_rows,
        "total_cols": n_cols,
        "missing_percentage": (isnull_sum.sum() / (n_rows * n_cols) * 100) if n_rows * n_cols > 0 else 0.0,
        "duplicate_rows": df.duplicated().sum(),
        "numeric_cols": len(numeric_cols),
        "categorical_cols": len(categorical_cols),
        "high_missing_cols": high_missing_cols,
        "constant_cols": constant_cols,
        "high_cardinality_cols": high_card_cols,
    }

def _smart_representative_sample(df: pd.DataFrame, n_samples: int = REP_SAMPLES, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    if len(df) <= n_samples:
        return df

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        strat_col = categorical_cols[0]
        try:
            grp = df.groupby(strat_col, group_keys=False)
            per_grp = max(1, n_samples // max(1, df[strat_col].nunique()))
            out = grp.apply(lambda x: x.sample(min(len(x), per_grp), random_state=random_state))
            return out.sample(n=min(n_samples, len(out)), random_state=random_state)
        except Exception:
            pass

    return _representative_csv_to_df(df, n_samples, REP_CLUSTERS, random_state)

def _representative_csv_to_df(df: pd.DataFrame, n_samples: int, n_clusters: int, random_state: int) -> pd.DataFrame:
    if len(df) <= n_samples:
        return df

    num_df = df.select_dtypes(include="number").copy()
    if num_df.empty or num_df.shape[1] == 0:
        return df.sample(n=n_samples, random_state=random_state)

    num_df = num_df.apply(pd.to_numeric, errors="coerce")
    num_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_df.dropna(axis=1, how="all", inplace=True)
    if num_df.empty:
        return df.sample(n=n_samples, random_state=random_state)

    num_df = num_df.fillna(num_df.mean(numeric_only=True))
    if len(num_df) < n_clusters + 1:
        return df.sample(n=n_samples, random_state=random_state)

    try:
        feat_values = num_df.to_numpy(dtype=float, copy=False)
        n_clusters_eff = min(n_clusters, len(num_df))

        if len(num_df) >= MINIBATCH_SWITCH_ROWS:
            km = MiniBatchKMeans(n_clusters=n_clusters_eff, random_state=random_state, batch_size=2048, n_init="auto", max_no_improvement=20)
        else:
            km = KMeans(n_clusters=n_clusters_eff, random_state=random_state, n_init="auto")

        clusters = km.fit_predict(feat_values)
        centers = km.cluster_centers_
    except Exception:
        return df.sample(n=n_samples, random_state=random_state)

    selected_idx = []
    feat_idx = num_df.index.to_numpy()
    unique_clusters = np.unique(clusters)
    per_cluster = [n_samples // len(unique_clusters)] * len(unique_clusters)
    for i in range(n_samples % len(unique_clusters)):
        per_cluster[i] += 1

    rng = np.random.default_rng(random_state)
    for c_i, c in enumerate(unique_clusters):
        mask = (clusters == c)
        idx_in_c = feat_idx[mask]
        if idx_in_c.size == 0:
            continue
        sub_vals = feat_values[mask]
        dists = np.linalg.norm(sub_vals - centers[c_i], axis=1)
        repr_local = idx_in_c[int(np.argmin(dists))]
        chosen = {repr_local}
        need_extra = per_cluster[c_i] - 1
        if need_extra > 0 and len(idx_in_c) > 1:
            others = idx_in_c[idx_in_c != repr_local]
            if len(others) > 0:
                extra = rng.choice(others, size=min(need_extra, len(others)), replace=False)
                chosen.update(extra)
        selected_idx.extend(chosen)

    if not selected_idx:
        return df.sample(n=n_samples, random_state=random_state)
    return df.loc[selected_idx]

def _enhanced_numeric_stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").astype("float64", copy=False).dropna()
    if s.empty:
        return dict(mean=np.nan, std=np.nan, min=np.nan, q25=np.nan, median=np.nan, q75=np.nan, max=np.nan, skewness=np.nan, kurtosis=np.nan)
    return {
        "mean": s.mean(),
        "std": s.std(),
        "min": s.min(),
        "q25": s.quantile(0.25),
        "median": s.median(),
        "q75": s.quantile(0.75),
        "max": s.max(),
        "skewness": stats.skew(s),
        "kurtosis": stats.kurtosis(s),
    }

def _enhanced_categorical_stats(series: pd.Series, top_k: int = 5) -> dict:
    vc = series.value_counts(dropna=True)
    return {
        "unique_count": series.nunique(dropna=True),
        "most_frequent": vc.index[0] if len(vc) > 0 else None,
        "top_categories": dict(vc.head(top_k)),
        "entropy": (stats.entropy(vc.values) if len(vc) > 1 else 0.0),
    }

# =========================== Prompt Generators ========================

def prompt_contextual_semantic(df: pd.DataFrame, dname: str, label: str) -> str:
    n_rows = len(df)
    numeric_mask = df.dtypes.apply(lambda dt: is_numeric_dtype(dt) and not is_bool_dtype(dt))

    schema_info = []
    for col in df.columns:
        s = df[col]
        is_num = bool(numeric_mask[col])
        null_pct = (s.isnull().sum() / n_rows) * 100 if n_rows > 0 else 0.0
        if is_num:
            st = _enhanced_numeric_stats(s)
            stat_txt = f"mean={st['mean']:.3g}, range=[{st['min']:.2g}, {st['max']:.2g}]"
        else:
            ct = _enhanced_categorical_stats(s)
            stat_txt = f"categories={ct['unique_count']}, mode='{ct['most_frequent']}'"
        schema_info.append(
            f"  {col}: {s.dtype} | {'target' if col == label else 'feature'} | {stat_txt} | missing={null_pct:.1f}%"
        )
    schema_text = "\n".join(schema_info)

    # strong correlations (non-bool numeric only)
    numeric_cols = df.select_dtypes(include="number").columns
    non_bool_numeric_cols = [c for c in numeric_cols if not is_bool_dtype(df[c].dtype)]
    relationships = []
    if len(non_bool_numeric_cols) > 1:
        try:
            num_df = df[non_bool_numeric_cols]
            if MAX_CORR_ROWS and len(num_df) > MAX_CORR_ROWS:
                num_df = num_df.sample(n=MAX_CORR_ROWS, random_state=RANDOM_STATE)
            num_df = num_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna(how="any").astype("float64", copy=False)
            if not num_df.empty and len(num_df) > 1:
                corr = num_df.corr(numeric_only=True)
                cols = corr.columns.tolist()
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        r = corr.iat[i, j]
                        if pd.notna(r) and abs(r) > 0.5:
                            relationships.append(f"{cols[i]} and {cols[j]}: correlation={r:.3f}")
        except Exception:
            pass
    rel_text = "\n".join([f"  • {x}" for x in relationships]) if relationships else "  • No strong correlations detected"

    tpl = f"""
        EMBED-ONLY PROMPT — GLOBAL CONTEXT FOR AUTOML PIPELINES: "{dname}"
    
    Produce a compact semantic summary of this dataset, suitable for a global context
    vector `ctx` consumed by a multi-stage pipeline controller. Do NOT explain model
    architectures, optimization procedures, or training algorithms; describe only the
    dataset and its statistical/structural properties.
    
    `ctx` is STATIC for the entire trajectory: it is computed once per dataset and
    reused across all states. It should capture global patterns about:
    - data types and distributions (numeric vs categorical, scale, skew),
    - missingness (overall severity and per-column patterns),
    - categorical structure (cardinality, imbalance),
    - correlations and redundancies between features,
    - task type (classification),
    - expected preprocessing difficulty,
    - opportunities for feature engineering and feature selection.
    
    The controller uses `(state, ctx)` via FiLM to make stage-specific
    decisions. Therefore `ctx` must highlight the signals most relevant to:
    
    Stage 0 — Logic Template Choice
    - First, choose ONE logic template from a small, finite, predefined set of templates
      (e.g., `comp.lpipelines`). Do NOT invent new templates.
    - Each template defines the fixed ORDER of pipeline stages, such as:
      [ImputerNum → ImputerCat → Encoder → Feature Preprocessing
       → Feature Engineering → Feature Selection].
    - `ctx` should make it easy to judge:
      - Missingness severity,
      - Categorical density,
      - Numeric distribution shape,
      - Feature redundancy / overlap.
    
    Stage 1 — Imputer
    - Identify which columns are numeric vs categorical.
    - Describe missingness patterns and ratios per column, especially extreme cases.
    
    Stage 2 — Encoder
    - Number of categorical columns.
    - For each categorical column: cardinality and imbalance of categories.
    
    Stage 3 — Feature Preprocessing
    - Scale ranges for numeric features.
    - Presence of outliers, skewness, and heavy tails in numeric features.
    
    Stage 4 — Feature Engineering
    - Strong linear and notable non-linear relationships among features.
    - Groups of features that form meaningful interactions.
    - Overall dimensionality and redundancy in the feature space.
    
    Stage 5 — Feature Selection
    - Constant or near-constant columns.
    - Very low-variance or highly duplicated columns and other obviously uninformative features.
    
    DATASET SUMMARY:
    - Dataset Name: {dname}
    - Target Column: "{label}"
    - Shape: {len(df):,} rows × {len(df.columns)} columns
    - Column Names: {", ".join(df.columns)}
    
    SCHEMA (per-column types, roles, missingness):
    {schema_text}
    CORRELATION SUMMARY (numeric-only, strong correlations):
    {rel_text}    
    FINAL OBJECTIVE:
    Encode the dataset’s essential global semantics such that:
    1) The logic template can be chosen correctly from the predefined finite set.
    2) Each stage can interpret `ctx` to guide its own primitive selection.
    3) The entire pipeline remains consistent with the dataset’s true distribution,
       without introducing new stages, components, or logic templates.
    Place the most decisive global insights NEAR THE END of the text, then terminate with:
    {SENTINEL_TOKEN}
    {SENTINEL_TOKEN}

    """
    return textwrap.dedent(tpl)





# ============================== IO layer ==============================

# 建议修改：
def _ensure_dirs():
    for style in PROMPT_STYLES: # 只遍历 PROMPT_STYLES
        os.makedirs(os.path.join(OUTPUT_ROOT, style), exist_ok=True)

def _preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # 将 object 列尝试识别为布尔或数值，提高后续统计稳健性
    for col in df.columns:
        if df[col].dtype == "object":
            s = df[col]
            vals = s.dropna().astype(str).str.lower().unique()
            if len(vals) > 0 and set(vals).issubset({"true", "false", "1", "0", "yes", "no"}):
                mapper = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
                df[col] = s.map(lambda x: mapper.get(str(x).lower(), x) if pd.notna(x) else x)
            else:
                numeric_series = pd.to_numeric(s, errors="coerce")
                if numeric_series.notna().sum() / len(df) > 0.5:
                    df[col] = numeric_series
    return df

def _find_csv_file(ds_dir: str) -> Optional[str]:
    direct = os.path.join(ds_dir, "data.csv")
    if os.path.isfile(direct):
        return direct
    for e in os.scandir(ds_dir):
        if e.is_file() and e.name.lower().endswith(".csv"):
            return e.path
    return None

def _detect_label(df: pd.DataFrame, dname: str) -> str:
    cfg = _load_dataset_config(CONFIG_PATH)
    for _, info in cfg.items():
        if info.get("dataset") == dname:
            label_key = info.get("label")
            if label_key is not None:
                try:
                    idx = int(label_key)
                    if 0 <= idx < len(df.columns):
                        return df.columns[idx]
                except ValueError:
                    if label_key in df.columns:
                        return label_key
    return df.columns[-1]  # fallback: 最后一列

def _write_prompt(out_dir: str, dname: str, payload: dict):
    out_path = os.path.join(out_dir, f"{dname}.csv")
    pd.DataFrame([payload]).to_csv(out_path, index=False, lineterminator="\n")

# ============================== Worker ================================

def _process_one_dataset(ds_dir: str, output_root: str):
    """
    返回: (prompt_count_by_style: dict, failure_or_None: dict|None, processed_flag: int)
    """
    prompt_count = {style: 0 for style in PROMPT_STYLES}
    dname = os.path.basename(ds_dir)

    csv_path = _find_csv_file(ds_dir)
    if csv_path is None:
        return prompt_count, {"name": dname, "reason": "No CSV file found", "details": f"Searched in: {ds_dir}"}, 0

    try:
        try:
            df = pd.read_csv(csv_path, low_memory=False, engine="pyarrow")
        except Exception:
            df = pd.read_csv(csv_path, low_memory=False)

        df = _preprocess_dataframe(df)

        if len(df) < 10:
            return prompt_count, {"name": dname, "reason": "Dataset too small", "details": f"Only {len(df)} rows (<10)"} , 0

        label = _detect_label(df, dname)
        failed_styles = []
        success_styles = []

        for style in PROMPT_STYLES:
            try:
                func = PROMPT_FUNC_MAP[style]
                prompt = func(df, dname, label)
                out_dir = os.path.join(output_root, style)
                payload = {
                    "table_name": dname,
                    "prompt": prompt,
                    "rows": len(df),
                    "cols": len(df.columns),
                    "label": label,
                    "style": style,
                }
                _write_prompt(out_dir, dname, payload)
                prompt_count[style] += 1
                success_styles.append(style)
            except Exception as e:
                failed_styles.append({"style": style, "error": str(e)})

        if failed_styles:
            return prompt_count, {
                "name": dname,
                "reason": "Partial style failures",
                "details": f"Failed styles: {[x['style'] for x in failed_styles]}",
                "success_styles": success_styles,
                "failed_styles": failed_styles,
            }, (1 if success_styles else 0)

        return prompt_count, None, 1

    except Exception as exc:
        return prompt_count, {"name": dname, "reason": "Processing error", "details": str(exc)}, 0



def main():
    args = parse_args()
    global DATA_ROOT, OUTPUT_ROOT, CONFIG_PATH
    
    BASE_DIR = "/data1/data_1/anoy/anoy_pipe"
    
    if args.dataset == "haipipe":
        print(">>> 正在处理：Haipipe (训练集)")
        DATA_ROOT = f"{BASE_DIR}/data/dataset"
        OUTPUT_ROOT = f"{BASE_DIR}/LLM/prompt_outputs/{args.dataset}"
        # CONFIG_PATH 保持不变
    elif args.dataset == "diffprep":
        print(">>> 正在处理：DiffPrep (测试集)")
        DATA_ROOT = f"{BASE_DIR}/data/diffprep_dataset"
        OUTPUT_ROOT = f"{BASE_DIR}/LLM/prompt_outputs/{args.dataset}"
        CONFIG_PATH = f"{BASE_DIR}/data/diffprep_dataset/classification_task_dic_diffprep.json"
    elif args.dataset == "deepline":
        print(">>> 正在处理：deepline (测试集)")
        DATA_ROOT = f"{BASE_DIR}/data/deepline_dataset"
        OUTPUT_ROOT = f"{BASE_DIR}/LLM/prompt_outputs/deepline"
        CONFIG_PATH = f"{BASE_DIR}/data/deepline_dataset/classification_task_dic_diffprep.json"


    print(f"当前数据集路径 (DATA_ROOT): {DATA_ROOT}")
    print(f"当前配置路径 (CONFIG_PATH): {CONFIG_PATH}")

    print("Prompt Generator for CtxPipe (Embed-Only)")
    print("=" * 60)
    _ensure_dirs()

    cfg = _load_dataset_config(CONFIG_PATH)
    dataset_names = [info["dataset"] for info in cfg.values()]

    total = len(dataset_names)
    print(f"发现数据集总数: {total}")
    print("正在处理中...(仅显示失败)\n")

    global_cnt = {style: 0 for style in PROMPT_STYLES}
    failed = []
    processed = 0

    if N_WORKERS <= 1 or total <= 1:
        for dname in dataset_names:
            ds_dir = os.path.join(DATA_ROOT, dname)
            pc, failure, ok = _process_one_dataset(ds_dir, OUTPUT_ROOT)
            processed += ok
            if failure:
                failed.append(failure)
            for k, v in pc.items():
                global_cnt[k] += v
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {
                ex.submit(_process_one_dataset, os.path.join(DATA_ROOT, dname), OUTPUT_ROOT): dname
                for dname in dataset_names
            }
            for fut in as_completed(futures):
                pc, failure, ok = fut.result()
                processed += ok
                if failure:
                    failed.append(failure)
                for k, v in pc.items():
                    global_cnt[k] += v

    print("=" * 60)
    print("处理结果汇总:")
    print(f"成功处理数据集: {processed}/{total}")
    print(f"失败数据集数量: {len(failed)}")

    print("\n各风格提示词生成数量:")
    for style, cnt in global_cnt.items():
        print(f"  {style:20}: {cnt:4d} prompts")

    total_prompts = sum(global_cnt.values())
    print(f"\n总提示词数量: {total_prompts}")
    if processed > 0:
        print(f"平均每数据集: {total_prompts / processed:.1f} 个提示词")

    if failed:
        print("\n" + "=" * 60)
        print("失败数据集详情:")
        print("=" * 60)
        for i, f in enumerate(failed, 1):
            print(f"\n{i}. 【{f['name']}】")
            print(f"   失败原因: {f['reason']}")
            print(f"   详细信息: {f['details']}")
            if 'success_styles' in f:
                print(f"   成功风格: {f['success_styles']}")
            if 'failed_styles' in f:
                print("   失败风格:")
                for s in f['failed_styles']:
                    print(f"     - {s['style']}: {s['error']}")
    else:
        print(f"\n🎉 所有 {total} 个数据集均处理成功！")


if __name__ == "__main__":
    main()

