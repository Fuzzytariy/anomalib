"""Run few-shot anomaly detection benchmarks for PaDiM, PatchCore and RegMem.

The script follows the experimental protocol defined in the research plan:

* evaluate multiple datasets (MVTec AD, MPDD, VisA or custom PCB datasets)
* sweep over few-shot support sizes (default 1/2/4 shots)
* optionally augment the support set with anomalous samples
* benchmark baseline (PaDiM, PatchCore) and proposed (RegMemFSAD) models

All experiment settings are defined in a YAML configuration. Each experiment run
produces image-level and pixel-level metrics using the anomalib Engine, measures
runtime and memory consumption, and stores the results in a CSV file.

Example::

    python tools/run_fsad_experiments.py --config examples/configs/experiments/few_shot.yaml
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    psutil = None

import torch

from anomalib.engine import Engine


@dataclass(slots=True)
class Scenario:
    """Definition of a support-set sampling scenario."""

    name: str
    anomaly_support: int = 0
    description: str | None = None
    models: set[str] | None = None


@dataclass(slots=True)
class ModelConfig:
    """Container for model instantiation details."""

    name: str
    class_path: str
    init_args: dict[str, Any]


@dataclass(slots=True)
class DatasetConfig:
    """Configuration for a dataset/category combination."""

    name: str
    datamodule_path: str
    init_args: dict[str, Any]
    categories: list[str]


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level configuration parsed from YAML."""

    output_dir: Path
    results_file: Path
    shots: list[int]
    repetitions: int
    random_seed: int
    scenarios: list[Scenario]
    models: list[ModelConfig]
    datasets: list[DatasetConfig]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/configs/experiments/few_shot.yaml"),
        help="Path to the experiment YAML configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for the output directory defined in the config.",
    )
    return parser.parse_args()


def _import_from_path(class_path: str) -> type[Any]:
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path:
        msg = f"Invalid class path '{class_path}'. Expected format '<module>.<ClassName>'."
        raise ValueError(msg)
    module = importlib.import_module(module_path)
    if not hasattr(module, class_name):
        msg = f"Class '{class_name}' not found in module '{module_path}'."
        raise ValueError(msg)
    return getattr(module, class_name)


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Convert Lightning metric outputs into serializable floats."""

    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            normalized[key] = float(value.detach().cpu().item())
        elif isinstance(value, (float, int)):
            normalized[key] = float(value)
    return normalized


def _sample_support_set(
    datamodule: Any,
    num_normal: int,
    num_anomalies: int,
    seed: int,
) -> None:
    """Restrict the training subset to a few-shot support set."""

    train_df = datamodule.train_data.samples.copy()
    normal_df = train_df[train_df.label_index == 0]
    if normal_df.empty:
        msg = "Training split does not contain normal samples to build a support set."
        raise RuntimeError(msg)

    sampled_normals = normal_df.sample(n=min(num_normal, len(normal_df)), random_state=seed)
    support_frames = [sampled_normals]

    if num_anomalies > 0:
        test_df = datamodule.test_data.samples
        anomaly_df = test_df[test_df.label_index == 1]
        if anomaly_df.empty:
            msg = (
                "Requested anomalous support samples but the test split does not contain any anomalous images."
            )
            raise RuntimeError(msg)
        sampled_anomalies = anomaly_df.sample(n=min(num_anomalies, len(anomaly_df)), random_state=seed)
        sampled_anomalies = sampled_anomalies.copy()
        sampled_anomalies["split"] = "train"
        support_frames.append(sampled_anomalies)

    support_df = pd.concat(support_frames, ignore_index=True)
    support_df.reset_index(drop=True, inplace=True)
    datamodule.train_data.samples = support_df

    total_support = len(support_df)
    datamodule.train_batch_size = max(1, min(datamodule.train_batch_size, total_support))


def _collect_memory_usage_mb() -> float | None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        return peak / (1024**2)

    if psutil is None:
        return None

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)


def _load_config(path: Path) -> ExperimentConfig:
    config = OmegaConf.load(path)

    output_dir = Path(config.get("output_dir", "results/few_shot"))
    results_file_cfg = config.get("results_file", "metrics.csv")
    results_file = Path(results_file_cfg)
    if not results_file.is_absolute():
        results_file = output_dir / results_file
    shots = list(config.get("shots", [1, 2, 4]))
    repetitions = int(config.get("repetitions", 1))
    random_seed = int(config.get("random_seed", 42))

    scenario_list: list[Scenario] = []
    for scenario in config.get("scenarios", []):
        scenario_list.append(
            Scenario(
                name=scenario["name"],
                anomaly_support=int(scenario.get("anomaly_support", 0)),
                description=scenario.get("description"),
                models=set(scenario.get("models") or []),
            )
        )

    model_list: list[ModelConfig] = []
    for model in config.get("models", []):
        model_list.append(
            ModelConfig(
                name=model["name"],
                class_path=model["class_path"],
                init_args={**(model.get("init_args") or {})},
            )
        )

    dataset_list: list[DatasetConfig] = []
    for dataset in config.get("datasets", []):
        dataset_list.append(
            DatasetConfig(
                name=dataset["name"],
                datamodule_path=dataset["datamodule"]["class_path"],
                init_args={**(dataset["datamodule"].get("init_args") or {})},
                categories=list(dataset.get("categories", [])),
            )
        )

    return ExperimentConfig(
        output_dir=output_dir,
        results_file=results_file,
        shots=shots,
        repetitions=repetitions,
        random_seed=random_seed,
        scenarios=scenario_list,
        models=model_list,
        datasets=dataset_list,
    )


def _ensure_iterable(obj: Iterable | Any) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, dict)):
        return list(obj)
    return [obj]


def run_experiments(config: ExperimentConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_cfg in config.datasets:
        datamodule_cls = _import_from_path(dataset_cfg.datamodule_path)

        for category in dataset_cfg.categories:
            for scenario in config.scenarios:
                for shot in config.shots:
                    for repetition in range(config.repetitions):
                        seed = config.random_seed + repetition
                        seed_everything(seed, workers=True)

                        for model_cfg in config.models:
                            if scenario.models and model_cfg.name not in scenario.models:
                                continue

                            init_args = {**dataset_cfg.init_args, "category": category}
                            init_args.setdefault("train_batch_size", max(shot, 1))
                            init_args.setdefault("eval_batch_size", 8)
                            datamodule = datamodule_cls(**init_args)
                            datamodule.seed = seed

                            datamodule.prepare_data()
                            datamodule.setup()

                            try:
                                _sample_support_set(
                                    datamodule=datamodule,
                                    num_normal=shot,
                                    num_anomalies=scenario.anomaly_support,
                                    seed=seed,
                                )
                            except RuntimeError as error:
                                rows.append(
                                    {
                                        "dataset": dataset_cfg.name,
                                        "category": category,
                                        "model": model_cfg.name,
                                        "scenario": scenario.name,
                                        "shot": shot,
                                        "repetition": repetition,
                                        "status": "failed",
                                        "error": str(error),
                                    }
                                )
                                continue

                            model_cls = _import_from_path(model_cfg.class_path)
                            model = model_cls(**model_cfg.init_args)

                            engine = Engine(logger=False, enable_progress_bar=False, accelerator="auto", devices=1)

                            if torch.cuda.is_available():
                                torch.cuda.reset_peak_memory_stats()

                            start_time = time.perf_counter()
                            test_metrics = engine.train(model=model, datamodule=datamodule)
                            elapsed = time.perf_counter() - start_time

                            metrics: dict[str, float] = {}
                            for metric_dict in _ensure_iterable(test_metrics):
                                metrics.update(_normalize_metrics(metric_dict))

                            metrics["inference_time_s"] = elapsed
                            memory_mb = _collect_memory_usage_mb()
                            if memory_mb is not None:
                                metrics["max_memory_mb"] = memory_mb

                            rows.append(
                                {
                                    "dataset": dataset_cfg.name,
                                    "category": category,
                                    "model": model_cfg.name,
                                    "scenario": scenario.name,
                                    "shot": shot,
                                    "repetition": repetition,
                                    "status": "ok",
                                    "metrics": json.dumps(metrics, sort_keys=True),
                                }
                            )

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(config.results_file, index=False)
    return dataframe


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    if args.output_dir is not None:
        original_output = config.output_dir
        config.output_dir = args.output_dir
        try:
            relative_results = config.results_file.relative_to(original_output)
        except ValueError:
            relative_results = Path(config.results_file.name)
        config.results_file = config.output_dir / relative_results

    config.output_dir.mkdir(parents=True, exist_ok=True)
    df = run_experiments(config)
    print(f"Saved results to {config.results_file} ({len(df)} rows)")


if __name__ == "__main__":
    main()
