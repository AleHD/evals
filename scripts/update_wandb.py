import collections
import statistics
import functools
import re
import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import wandb


def get_log(infos: List[dict], tasks_cfg: dict, language_cfg: dict[str, list[str]],
            dimensions_cfg: dict[str, list[str]]) -> Dict[str, float]:

    def agg(log: dict[str, dict[str, float]], prefix: str, tasks_to_agg: list[str], warn: bool = True):
        missing = set(log) - set(tasks_to_agg)
        if len(missing) > 0:
            if warn:
                print("WARNING! Macro aggregation for", prefix, "not available. Missing:", sorted(missing))
            return

        for metric in filter(lambda metric: "stderr" not in metric, all_metrics):
            values = [log[taskname][metric] for taskname in tasks_to_agg if metric in log[dataname]]
            if len(values) > 0:
                log[f"{prefix}.macro"][metric] = statistics.mean(values)

    # Aggregate raw info.
    groups = {}
    results = {}
    all_metrics = set()
    for info in infos:
        groups.update(info["group_subtasks"])
        results.update(info["results"])

    # Prepare final logs.
    log = collections.defaultdict(dict)
    for dataname, details in results.items():
        for metricname, val in details.items():
            if metricname == "alias" or val == "N/A":
                continue
            assert isinstance(val, float), val
            metricname, _ = metricname.split(",")  # for some reason it is always acc,none so we remove the none.
            all_metrics.add(metricname)
            log[dataname][metricname] = val

    # Filter manual harness groups.
    for name in ["swissai_eval", "english", "multilingual", "english_pt1", "english_pt2", "multilingual_pt1", "multilingual_pt2"]:
        if name in log:
            del log[name]
    remaining = list(log)
    for key in remaining:
        if any(key.startswith(dimension) for dimension in dimensions_cfg):
            del log[key]

    # Now that we have all the "leaf task groups" we can do three aggregations:
    # {language_group}, {dimension}.{language_group} and {dimension}.{language}.
    # Let's start with the {language_group} agg.
    for lang_group_name, lang_group in tasks_cfg["language_groups"].items():
        nested = [tasks for lang, tasks in language_cfg.items()
                  if lang in lang_group]
        tasks_to_agg = functools.reduce(list.__add__, nested)
        agg(log, lang_group_name, tasks_to_agg)

    # Agregate {dimension}.{language_group}.macro.
    for lang_group_name, lang_group in tasks_cfg["language_groups"]:
        nested = [tasks for lang, tasks in language_cfg.items() if lang in lang_group]
        lang_tasks = functools.reduce(list.__add__, nested)
        for dimension, dim_tasks in dimensions_cfg.items():
            tasks_to_agg = list(set(lang_tasks) & set(dim_tasks))
            agg(log, f"{dimension}.{lang_group_name}", tasks_to_agg)

    # Finally, {dimension}.{language}
    for lang_name, lang_tasks in language_cfg.items():
        for dimension, dim_tasks in dimensions_cfg.items():
            tasks_to_agg = list(set(lang_tasks) & set(dim_tasks))
            agg(log, f"{dimension}.{lang_name}", tasks_to_agg)

    # Finally, prepare wandb format.
    wandb_log = {}
    for dataname, details in log.items():
        for metric, value in details.items():
            wandb_log[f"{dataname}/{metric}"] = value
    return wandb_log


def get_history(name: str) -> Dict[int, Dict[str, float]]:
    api = wandb.Api()
    try:
        run = api.run(f"{api.default_entity}/{os.environ['WANDB_PROJECT']}/{name}")
    except wandb.errors.errors.CommError:  # Run not found.
        return {}
    history = collections.defaultdict(dict)
    for row in run.scan_history():
        row = {key: value for key, value in row.items() if not key.startswith("_")}
        history[row["ConsumedTokens"]] = row
    return history


def main(logs_root: Path, name: Optional[str], it: Optional[int],
         cfg: Path):

    with open(cfg/"languages.json") as f:
        languages_cfg = json.load(f)
    with open(cfg/"tasks.json") as f:
        tasks_cfg = json.load(f)
    with open(cfg/"dimensions.json") as f:
        dimensions_cfg = json.load(f)

    for lang_group in tasks_cfg["language_groups"].values():
        for lang in lang_group:
            assert lang in languages_cfg, lang
    lang_tasks = set(functools.reduce(list.__add__, languages_cfg.values()))
    dim_tasks = set(functools.reduce(list.__add__, dimensions_cfg.values()))
    assert lang_tasks == dim_tasks, f"{sorted(lang_tasks ^ dim_tasks)}"

    tasks_cfg["language_groups"]["global"] = list(languages_cfg)
    tasks_cfg["language_groups"]["multilingual"] = list(set(languages_cfg) - {"English"})

    # Grab each possible log and update wandb run.
    # First, iterate model names.
    latest_logs = {}
    for p1 in filter(lambda p: name is None or name == p.name, logs_root.iterdir()):
        print("Updating path", p1)
        history = get_history(p1.name)  # Get already pushed information.
        with wandb.init(id=p1.name, name=p1.name) as run:
            run.define_metric("ConsumedTokens")
            run.define_metric("*", step_metric="ConsumedTokens")
            # Now iterate all iterations for this name.
            for p2 in p1.iterdir():
                current_it = int(re.match("^iter_([0-9]+)$", p2.name).group(1))
                # Skip if specified --it doesn't match currently iterated it.
                if it is not None and it != current_it:
                    continue
                print("Updating iteration", current_it)

                with open(p2/"consumed_tokens.txt") as f:
                    consumed_tokens = int("".join(f).strip())

                # Get all results.json harness logs.
                results = []
                for path in sorted(p2.glob("harness/eval_*/*/results*.json")):
                    with open(path) as f:
                        results.append(json.load(f))

                if len(results) > 0:
                    log = get_log(results, tasks_cfg, language_cfg, dimensions_cfg)
                    log.update({"ConsumedTokens": consumed_tokens, "OptStep": current_it})
                    continue
                    sublog = {k: v for k, v in log.items() if "macro/acc" in k}
                    # Update log if needed.
                    if consumed_tokens in history:
                        if "eval_table" in history[consumed_tokens]:
                            del history[consumed_tokens]["eval_table"]
                        if log == history[consumed_tokens]:
                            print("Exact log already matches wandb! Ignoring entry to avoid pushing duplicates")
                        else:
                            print(sorted(set(history[consumed_tokens]) - set(log)))
                            print("Important! wandb log at current iteration already found, but differs. Updating")
                            run.log(log)
                            print("Logged sucessful:", sublog)
                    else:
                        run.log(log)
                        print("Logged sucessful:", sublog)

                    # Update all_logs so we can build the table after this big loop.
                    if p1.name not in latest_logs or latest_logs[p1.name]["ConsumedTokens"] < consumed_tokens:
                        latest_logs[p1.name] = log
                else:
                    print("No logs found!")
                print()

    # Build and push the table.
    # We need `it` to be None to ensure that the logs on `latest_logs` actually
    # belong to the latest known iteration.
    show_in_table = tasks_cfg["show_in_table"]
    if it is None:
        for name, log in filter(lambda t: set(show_in_table) <= set(t[1]),
                                latest_logs.items()):
            print("Updating table for model", name)
            sublog = {"Model": name}
            sublog.update({task: log[task] for task in ["ConsumedTokens"] + show_in_table})
            df = pd.DataFrame([sublog])
            with wandb.init(id=name, name=name) as run:
                run.log({"eval_table": wandb.Table(dataframe=df), "ConsumedTokens": log["ConsumedTokens"]})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("logs_root", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--it", type=int)
    parser.add_argument("--cfg", type=Path, default=Path("configs"))
    args = parser.parse_args()
    main(**vars(args))
