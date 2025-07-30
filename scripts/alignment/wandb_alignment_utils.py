"""
Shared utilities for W&B alignment evaluation scripts.
Contains common functions for collecting, processing, and uploading evaluation results.
"""

import json
import wandb
from pathlib import Path
from typing import Dict, List


def collect_results(eval_dir: Path) -> List[Dict]:
    """Collect all results from an evaluation directory."""
    if not eval_dir.exists():
        return []
    
    results = []
    for result_file in eval_dir.glob("**/results*.json"):
        try:
            with open(result_file) as f:
                results.append(json.load(f))
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read {result_file}: {e}")
    return results


def flatten_results(results: List[Dict]) -> Dict[str, float]:
    """Flatten results into a single dictionary with metric names as keys."""
    flat = {}
    for res in results:
        if "results" not in res:
            continue
        for task, metrics in res["results"].items():
            for metric_name, value in metrics.items():
                if metric_name == "alias" or value in ["N/A", " ", None]:
                    continue
                try:
                    metric = metric_name.split(",")[0]
                    flat[f"{task}/{metric}"] = float(value)
                except (ValueError, TypeError):
                    continue
    return flat


def create_wandb_table(run_id: str, main_log_data: dict) -> wandb.Table:
    """Create a W&B table for a single model run."""
    columns = ["model"] + list(main_log_data.keys())
    table_data = [[run_id] + list(main_log_data.values())]
    return wandb.Table(data=table_data, columns=columns)


def find_all_eval_dirs(logs_root: Path, model_name: str) -> List[Path]:
    """Find all evaluation directories for a given model, sorted from oldest to newest."""
    model_log_dir = logs_root / model_name
    harness_dirs = list(model_log_dir.glob("harness/eval_*"))
    return sorted(harness_dirs, key=lambda x: x.name)


def _upload_to_wandb(entity: str, project: str, run_id: str, run_name: str, log_data: Dict[str, float], main_metrics: List[str]):
    """Common function to upload data to W&B."""
    wandb.login()
    
    # Main log_data to only include keys that start with eval names from the file
    main_log_data = {}
    for eval_metric in main_metrics:
        if eval_metric in log_data:
            main_log_data[eval_metric] = log_data[eval_metric]
    
    run_id_suffix = "-001"
    
    with wandb.init(
        id=run_id + run_id_suffix,
        resume="allow",
        entity=entity,
        project=project,
        name=run_name,
    ) as run:
        run.log({"main_results": create_wandb_table(run_name, main_log_data)})
        run.log(log_data)
        print(f"Logged to WandB for {run_name}: {len(log_data)} entries")


def upload_multi_model_results(entity: str, project: str, all_results: Dict[str, Dict[str, float]], main_metrics: List[str]):
    """Upload results from one or multiple models to W&B, each as a separate run."""
    model_count = len(all_results)
    print(f"Uploading {model_count} model(s) to W&B")
    
    for model_name, log_data in all_results.items():
        print(f"\nUploading {model_name}...")
        _upload_to_wandb(entity, project, model_name, model_name, log_data, main_metrics)
    
    print(f"\nSuccessfully uploaded {model_count} model(s) to W&B project {project}")
