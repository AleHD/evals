#!/usr/bin/env python3
"""
Script to scan all evaluation logs and update W&B with results from all models.
This script goes through all existing runs in the log directory and updates a W&B table.
"""

from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, List

from .wandb_alignment_utils import (
    collect_results, 
    flatten_results, 
    find_all_eval_dirs, 
    upload_multi_model_results
)


# Define all models that we want to process
ALL_MODELS = [
    # Apertus RLVR reasoning models
    "Apertus3-8B_iter_1678000-tulu3-sft-RLVR-105",
    "Apertus3-8B_iter_1678000-tulu3-sft-RLVR-450",
    "Apertus3-8B_iter_1678000-tulu3-sft-RLVR-560",
    
    # Apertus base pretrained models
    "Apertus8B-tokens7.04T-it1678000",
    "Apertus8B-tokens7.4T-it1728000",
    "Apertus70B-tokens15T-it1155828",
    
    # Apertus SFT trained models
    "Apertus-8B-SFT",
    "Apertus-70B-15T-it1155828--Tulu3-SFT",
    "Apertus-8B-7.04T-it1678000--Tulu3-SFT-tulu",
    "Apertus-8B-7.04T-it1678000--Tulu3-SFT-tulu_special_token",
    "Apertus8B-tokens7.4T-it1728000-ademamix-swissai-tulu-3-sft-0225",
    
    # EuroLLM models
    "EuroLLM-9B",
    "EuroLLM-22B-Preview",
    
    # Gemma models
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    
    # K2 models
    "K2-Chat",
    
    # Llama models
    "Llama-3.1-8B-Instruct",
    "Llama-3.3-70B-Instruct",
    
    # OLMo models (base)
    "OLMo-2-0325-32B",
    "OLMo-2-1124-7B-Instruct",
    
    # OLMo models (fine-tuned variants)
    "OLMo-2-0325-32B-DPO",
    "OLMo-2-0325-32B-Instruct",
    "OLMo-2-0325-32B-SFT",
    
    # Qwen 2.5 models
    "Qwen2.5-7B",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct",
    "Qwen2.5-32B-Instruct",
    "Qwen2.5-72B-Instruct",
    
    # Qwen 3 models
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    
    # SmolLM models
    "SmolLM3-3B",
]


def load_main_metrics() -> List[str]:
    """Load main metrics from config file."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "alignment" / "tasks_english_main_table.txt"
    with open(config_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def scan_all_models(logs_root: Path) -> Dict[str, Dict[str, float]]:
    """Scan all models and collect their results."""
    all_results = {}
    print(f"Scanning logs in: {logs_root}")

    # Process models that are in our defined list
    processed_count = 0
    for model_name in ALL_MODELS:
        print(f"Processing {model_name}")
        eval_dirs = find_all_eval_dirs(logs_root, model_name)
        
        # Collect results from all directories, newer ones overwrite older ones
        combined_results = {}
        for eval_dir in eval_dirs:
            results = collect_results(eval_dir)
            flattened = flatten_results(results)
            
            # Update combined results, newer metrics overwrite older ones
            combined_results.update(flattened)
            print(f"  + {eval_dir.name}: {len(flattened)} metrics")
        
        all_results[model_name] = combined_results
        processed_count += 1
        print(f"  ✓ Total: {len(combined_results)} metrics")
    
    print(f"Successfully processed {processed_count} models")
    return all_results


def main():
    parser = ArgumentParser(description="Scan all evaluation logs and update W&B with model results")
    parser.add_argument("--entity", type=str, required=True, help="W&B entity name")
    parser.add_argument("--project", type=str, required=True, help="W&B project name")
    parser.add_argument("--logs_root", type=Path, default="/iopsstor/scratch/cscs/ihakimi/eval-logs", 
                       help="Root directory containing all evaluation logs")
    parser.add_argument("--main_metrics", nargs='+', type=str, default=None,
                       help="List of main metrics for the summary table (defaults to config file)")
    parser.add_argument("--dry_run", action="store_true", help="Just scan and print results without uploading to W&B")
    
    args = parser.parse_args()
    
    # Load main metrics from config if not provided
    if args.main_metrics is None:
        args.main_metrics = load_main_metrics()
    
    # Scan all models
    all_results = scan_all_models(args.logs_root)
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Found results for {len(all_results)} models:")
    for model_name, metrics in all_results.items():
        available_main_metrics = sum(1 for metric in args.main_metrics if metric in metrics)
        print(f"  {model_name}: {len(metrics)} total metrics, {available_main_metrics}/{len(args.main_metrics)} main metrics")
    
    if args.dry_run:
        print("\nDry run completed. No data uploaded to W&B.")
        return
    
    # Upload to W&B
    upload_multi_model_results(args.entity, args.project, all_results, args.main_metrics)


if __name__ == "__main__":
    main()
