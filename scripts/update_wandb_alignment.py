import json
from pathlib import Path
from argparse import ArgumentParser
import wandb



def collect_results(latest_eval_dir: Path):
    if latest_eval_dir is None:
        return []
    
    results = []
    
    # Only collect results from the latest eval directory
    for result_file in latest_eval_dir.glob("**/results*.json"):
        with open(result_file) as f:
            results.append(json.load(f))
    return results


def flatten_results(results):
    flat = {}
    for res in results:
        for task, metrics in res["results"].items():
            for metric_name, value in metrics.items():
                if metric_name == "alias" or value in ["N/A", " "]:
                    continue
                metric = metric_name.split(",")[0]
                flat[f"{task}/{metric}"] = float(value)
    return flat


def create_wandb_table(run_id: str, main_log_data: dict):
    columns = ["model"] + list(main_log_data.keys())
    table_data = [[run_id] + list(main_log_data.values())]
    return wandb.Table(data=table_data, columns=columns)


def upload_run(entity: str, project: str, run_id: str, main_metrics: list, logs_root: Path):
    wandb.login()

    print(f"Uploading {run_id}, iteration: {logs_root.name}")

    results = collect_results(logs_root)
    if not results:
        print(f"No result files found for {run_id}")
        return

    log_data = flatten_results(results)
    print(f"Collected {len(log_data)} metrics for {run_id}")
    print(f"Log data keys: {list(log_data.keys())}")

    # Main log_data to only include keys that start with eval names from the file
    main_log_data = {}
    for eval_metric in main_metrics:
        if eval_metric in log_data:
            main_log_data[eval_metric] = log_data[eval_metric]
        else:
            print(f"Warning: Metric {eval_metric} not found in log data for {logs_root.name}")

    run_id_suffix = "-001" # this is used incase we need to delete all experiments for some reason and then reupload new ones

    with wandb.init(
        id=run_id+run_id_suffix,
        resume="allow",
        entity=entity,
        project=project,
        name=run_id,
    ) as run:
        run.log({"main_results": create_wandb_table(run_id, main_log_data)})
        run.log(log_data)
        print(f"Logged to WandB for {run_id}: {len(log_data)} entries in table format")


def main(entity: str, project: str, name: str, main_metrics: Path, logs_root: Path):
    upload_run(entity, project, name, main_metrics, logs_root)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--name", type=str, required=True, help="Name of the model")
    parser.add_argument("--main_metrics", nargs='+', type=str, required=True, help="List of metrics for main table")
    parser.add_argument("--logs_root", type=Path, required=True, help="Root directory containing evaluation logs")
    args = parser.parse_args()

    main(entity=args.entity, project=args.project, name=args.name, main_metrics=args.main_metrics, logs_root=args.logs_root)
