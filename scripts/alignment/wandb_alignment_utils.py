"""
Shared utilities for W&B alignment evaluation scripts.
Contains common functions for collecting, processing, and uploading evaluation results.
"""

import json
import wandb
from pathlib import Path
from typing import List

from .data_structures import Sample, Metric, Task, ModelEvaluation


def create_model_evaluation_from_results(model_name: str, eval_dir: Path, max_samples: int = 10) -> ModelEvaluation:
    """Create a ModelEvaluation directly from evaluation directory."""
    
    tasks = []
    
    # Process metrics from results and create tasks immediately
    result_files = list(eval_dir.glob("**/results_*.json"))
    
    assert len(result_files) == 1, f"Expected exactly one results file, found {len(result_files)} in {eval_dir}"
    
    result_file = result_files[0]
    with open(result_file) as f:
        res = json.load(f)
    
    # Extract timestamp from results filename: results_2025-07-26T00-35-42.178646.json
    timestamp = result_file.stem.replace("results_", "")

    for task_name, metrics in res["results"].items():
        # Create Metric objects for this task
        task_metrics = []
        for metric_name, value in metrics.items():
            if metric_name == "alias" or value in ["N/A", " ", None]:
                continue

            metric = metric_name.split(",")[0]  # Clean metric name
            task_metrics.append(Metric(name=metric, score=float(value)))
        
        # Load corresponding samples for this task using exact filename
        task_samples = []
        sample_files = list(eval_dir.glob(f"**/samples_{task_name}_{timestamp}.jsonl"))
        assert len(sample_files) == 1, f"Expected exactly one sample file for task {task_name}, found {len(sample_files)}"
        with open(sample_files[0], 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                sample_data = json.loads(line.strip())
                task_samples.append(Sample(sample_data=sample_data))
        
        # Create Task object immediately
        tasks.append(Task(
            task_name=task_name,
            metrics=task_metrics,
            samples=task_samples
        ))
    
    return ModelEvaluation(model_name=model_name, tasks=tasks)


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


def upload_multi_model_results(entity: str, project: str, model_evaluations: List[ModelEvaluation], main_metrics: List[str]):
    """Upload results from ModelEvaluation data structures to W&B, each as a separate run."""
    model_count = len(model_evaluations)
    print(f"Uploading {model_count} model(s) to W&B")
    
    for model_eval in model_evaluations:
        print(f"\nUploading {model_eval.model_name}...")
        print(f"  - {model_eval.total_metrics_count} metrics across {len(model_eval.tasks)} tasks")
        print(f"  - {model_eval.total_samples_count} samples")
        
        # Upload to W&B with structured samples
        _upload_to_wandb_with_model_eval(entity, project, model_eval, main_metrics)
    
    print(f"\nSuccessfully uploaded {model_count} model(s) to W&B project {project}")


def _upload_to_wandb_with_model_eval(entity: str, project: str, model_eval: ModelEvaluation, main_metrics: List[str]):
    """Upload ModelEvaluation data to W&B with structured samples."""
    wandb.login()
    
    # Get flattened metrics for W&B logging
    log_data = model_eval.get_flattened_metrics()
    
    # Main log_data to only include keys that start with eval names from the file
    main_log_data = {}
    for eval_metric in main_metrics:
        if eval_metric in log_data:
            main_log_data[eval_metric] = log_data[eval_metric]
    
    run_id_suffix = "-001"
    
    with wandb.init(
        id=model_eval.model_name + run_id_suffix,
        resume="allow",
        entity=entity,
        project=project,
        name=model_eval.model_name,
    ) as run:
        run.log({"main_results": create_wandb_table(model_eval.model_name, main_log_data)})
        run.log(log_data)
        
        # Upload samples as a table directly from the structured data
        for task in model_eval.tasks:
            if not task.samples:
                print(f"  - No samples for task {task.task_name}, skipping")
                continue

            samples_table = upload_structured_samples_as_table(task)
            run.log({f"samples/{model_eval.model_name}/{task.task_name}": samples_table})
            
        print(f"Logged to WandB for {model_eval.model_name}: {len(log_data)} entries")


def upload_structured_samples_as_table(task: Task):
    """Create and return a W&B table with samples from a single task."""
    all_rows = [_flatten_dict(sample.sample_data) for sample in task.samples]
    columns = list(all_rows[0].keys())
    table_data = [[row.get(col) for col in columns] for row in all_rows]
    return wandb.Table(data=table_data, columns=columns)


def _flatten_dict(d, parent_key='', sep='/'):
    """Recursively flatten a nested dictionary with separator between keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to strings for table compatibility
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


