#!/usr/bin/env python3
"""
aggregate_results.py – Compute regional aggregates for Harness eval outputs.

Example
-------
python aggregate_results.py \
    --results out/results.json \
    --benchmarks src/language_benchs.json \
    --output out/results_aggregated.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

# ---------------------------- language groups ---------------------------- #

SWISS_LANGS: list[str] = ["German", "French", "Italian", "Romansh"]
EU_LANGS: list[str] = [
    "Albanian", "Armenian", "Basque", "Belarusian", "Bulgarian", "Catalan",
    "Croatian", "Czech", "Danish", "Dutch", "English", "Estonian", "Finnish",
    "French", "Georgian", "German", "Greek", "Hungarian", "Italian",
    "Lithuanian", "North Macedonian", "Polish", "Portuguese", "Romanian",
    "Romansh", "Russian", "Serbian", "Slovak", "Spanish", "Swedish",
    "Ukrainian",
]

# ---------------------------- helpers ---------------------------- #

Number = float | int


def is_number(v: Any) -> bool:
    """Return True if *v* can be losslessly cast to float."""
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def aggregate_metrics(task_names: Iterable[str], results: Dict[str, Dict[str, Any]]) -> Dict[str, Number | str]:
    """
    Compute the mean for every metric that appears in *task_names*.

    Non-numeric or missing values are ignored. If every value for a metric is
    missing, the metric is set to 'N/A'.
    """
    aggregated: Dict[str, List[Number]] = {}

    for task in task_names:
        if task not in results:
            continue
        for metric, val in results[task].items():
            if is_number(val):
                aggregated.setdefault(metric, []).append(float(val))

    final: Dict[str, Number | str] = {}
    for metric, vals in aggregated.items():
        if vals:
            final[metric] = mean(vals)
        else:
            final[metric] = "N/A"

    return final


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logging.error("Cannot read %s – %s", path, exc)
        sys.exit(1)


def save_json(obj: Any, path: Path) -> None:
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4, ensure_ascii=False)
    except OSError as exc:
        logging.error("Cannot write %s – %s", path, exc)
        sys.exit(1)


# ---------------------------- main ---------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Add aggregate results to a Harness output file.")
    p.add_argument("--results", required=True, type=Path,
                   help="Path to the existing Harness results JSON.")
    p.add_argument("--benchmarks", required=True, type=Path,
                   help="language_bench.json containing the set of ALL_LANGUAGES.")
    p.add_argument("--output", required=True, type=Path,
                   help="Where to write the augmented JSON.")
    p.add_argument("-q", "--quiet", action="store_true", help="Silence info logs.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    logging.info("Reading input files…")
    harness_file: dict[str, Any] = load_json(args.results)
    results: Dict[str, Dict[str, Any]] = harness_file["results"]

    language_bench = load_json(args.benchmarks)
    all_languages: list[str] = list(language_bench.keys())

    lang_groups: dict[str, list[str]] = {
        "swiss": SWISS_LANGS,
        "eu": EU_LANGS,
        "global": all_languages,
    }

    for dimension in ("general_abilities", "factual_agnostic", "factual_regional"):
        for tag, langs in lang_groups.items():
            tasks = [f"{dimension}_{lang.lower()}" for lang in langs]
            agg = aggregate_metrics(tasks, results)
            task_name = f"{dimension}_{tag}"
            agg["alias"] = task_name
            results[task_name] = agg
            logging.info("Added %s (%d tasks, %d metrics).",
                         task_name, len(tasks), len(agg) - 1)

    harness_file["results"] = results
    save_json(harness_file, args.output)
    logging.info("Wrote augmented results → %s", args.output)


if __name__ == "__main__":
    main()
