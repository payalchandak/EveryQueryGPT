import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import inflect

inflect = inflect.engine()
import polars as pl
from omegaconf import DictConfig, OmegaConf

from .config import *
from .dataset_polars import Dataset


def query_task_DAG(events_df: pl.DataFrame, task_dag) -> pl.DataFrame:
    exprs = {}
    nodes, relations = task_dag
    parents = {}

    exprs_to_run = {}

    for node, children in relations:
        if type(exprs[node]) is pl.Expr:
            if node in parents:
                valid_expr = exprs[node] & (pl.col('timestamp') >= exprs[parents[node]])
            else:
                valid_expr = exprs[node]

            exprs[node] = pl.when(valid_expr, pl.col('timestamp'), pl.lit(None))
        else:
            exprs[node] = dereference(node, exprs)
        for child in children: parents[child] = node



nsample_task_nodes = {
    "trigger.event": pl.col('admission')
    "gap.start": "${trigger.event}"
    "gap.end": "${trigger.event} + 2d"
    "input.end": "${trigger.event} + 1d"
    "target.start": "${gap.end}"
    "target.end": pl.col('discharge') | pl.col('death')
}

sample_task_DAG = [
    ("trigger.event", ["gap.start", "input.end"]),
    ("input.end", [])
    ("gap.start", ["gap.end"]),
    ("gap.end", ["target.start"]),
    ("target.start", ["target.end"]),
    ("target.end", [])
]

def query_dataset(D: Dataset, T: TaskConfig) -> pl.DataFrame:
    # 1. Get the event predicates
    root_event_predicates = T.get_event_predicates()
    root_event_value_fns = T.get_event_value_fns()

    # 2. Get the static variables needed
    static_variables = T.get_static_variables()

    # 3. Select events, subject_IDs, and timestamps
    static_df = D.subjects_df.lazy().select("subject_id", *static_variables)

    event_predicates_df = (
        D.dynamic_measurements_df
        .lazy()
        .join(D.events_df.lazy().select("event_id", "event_type"), on="event_id")
        .group_by("event_id")
        .agg(**root_event_predicates, **root_event_value_fns)
    )

    events_df = (
        D.events_df.lazy()
        .join(static_df, on='subject_id', how='inner')
        .join(event_predicates_df, on='event_id', how='inner')
        .select(
            'subject_id', *static_variables, 'timestamp',
            *root_event_predicates.keys(), *root_event_value_fns.keys()
        )
    )

    # Fit event windows
    #window_endpoint_DAG = T.get_window_endpoint_DAG()
    window_timepoints_values, window_valid = T.get_query()
    events_df = (
        events_df
        .with_column(**window_timepoints_values)
        .filter(window_valid)
    )




def parse_task_cfg(task_cfg: DictConfig) -> dict[str, TaskConfig] | TaskConfig:
    task_fields = {f.name for f in dataclasses.fields(TaskConfig)}
    direct_task_kwargs = {k: v for k, v in task_cfg.items() if k in task_fields}
    extra_kwargs = {k: v for k, v in task_cfg.items() if k not in task_fields}

    base_task_cfg = TaskConfig(**direct_task_kwargs)
    if not extra_kwargs:
        return base_task_cfg

    out = {}
    for k, v in extra_kwargs.items():
        v = parse_task_cfg(v)
        match v:
            case dict():
                for k2, v2 in v.items():
                    out[f"{k}/{k2}"] = base_task_config.merge(v2)
            case TaskConfig():
                out[k] = base_task_config.merge(v)
            case _:
                raise ValueError(f"Invalid task config: {v}")
    return out

def safe_merge(
    container: dict[str, Any], updates_dict: dict[str, Any], str_key: str | None = None
) -> dict[str, Any]:
    for key, val in updates_dict:
        if key in container:
            old_val = container[key]
            k_str = key if str_key is None else f"{str_key}.{key}"
            if isinstance(val, dict) and isinstance(old_val, dict):
                safe_merge(old_val, val)
            elif val == old_val:
                print(f"WARNING: duplicate key {k_str} encountered with shared value {val}!")
            else:
                raise ValueError(f"Duplicate key {k_str} encountered with differing values {val} vs. {old_val}")
        else: container[key] = val

def expand_task_cfg(task_cfg: DictConfig) -> DictConfig:
    out = defaultdict(dict)
    for key, value in task_cfg.items():
        parts = key.split(".")
        key = parts[0]
        for part in parts[1::-1]:
            value = {part: value}

        safe_merge(out[key], expand_task_cfg(value))
    return DictConfig(**out)


@hydra.main(version_base=None, config_path="../configs", config_name="task_base")
def main(cfg: DictConfig):
    cfg = hydra.utils.instantiate(cfg, _convert_="all")

    task_df_dir = Path(cfg["dataset_dir"]) / "task_dfs"

    # parents=False because dataset_dir must already exist
    task_df_dir.mkdir(exist_ok=True, parents=False)

    cfg_fp = task_df_dir / "hydra_config.yaml"
    OmegaConf.save(cfg, cfg_fp)

    tasks = parse_task_cfg(cfg["tasks"])

    task_configs = parse_task_cfg(tasks)

    for task_name, task_cfg in tasks.items():
        task_cfg_fp = task_df_dir / f"{task_name}_config.yml"
        OmegaConf.save(task_cfg, task_cfg_fp)

        task_df = build_task_df(task_cfg)
        task_fp = task_df_dir / f"{task_name}.parquet"
        task_fp.parent.mkdir(exist_ok=True, parents=True)

        task_df.to_parquet(task_fp)


if __name__ == "__main__":
    main()
