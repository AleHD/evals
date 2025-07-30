from pathlib import Path
from argparse import ArgumentParser

from .wandb_alignment_utils import upload_multi_model_results, collect_results, flatten_results


def main(entity: str, project: str, name: str, main_metrics: list, logs_root: Path):
    print(f"Uploading {name}, iteration: {logs_root.name}")
    
    results = collect_results(logs_root)
    log_data = flatten_results(results)
    print(f"Collected {len(log_data)} metrics for {name}")
    print(f"Log data keys: {list(log_data.keys())}")
    
    # Use upload_multi_model_results with a single model
    single_model_results = {name: log_data}
    upload_multi_model_results(entity, project, single_model_results, main_metrics)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--name", type=str, required=True, help="Name of the model")
    parser.add_argument("--main_metrics", nargs='+', type=str, required=True, help="List of metrics for main table")
    parser.add_argument("--logs_root", type=Path, required=True, help="Root directory containing evaluation logs")
    args = parser.parse_args()

    main(entity=args.entity, project=args.project, name=args.name, main_metrics=args.main_metrics, logs_root=args.logs_root)
