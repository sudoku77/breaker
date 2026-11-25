from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml

from breaker.analysis.export import write_comparison_csv
from breaker.analysis.plots import plot_comparison
from breaker.analysis.compare import comparison_metrics


PROJECT_DIR = Path(__file__).resolve().parent


def load_config(path: str | Path = PROJECT_DIR / "configs" / "default.yaml") -> dict:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["usd_path"] = str((config_path.parent / config["usd_path"]).resolve())
    config["output_dir"] = str((PROJECT_DIR / config["output_dir"]).resolve())
    return config


def main() -> None:
    config = load_config()
    from breaker.simulation.isaac_collector import run_isaac_simulation

    result = run_isaac_simulation(config)
    print(result)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config["output_dir"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(output_dir)
    write_comparison_csv(output_dir / "comparison.csv", result)
    plot_comparison(output_dir, result)
    metadata = {
        "run_id": run_id,
        "config": config,
        **result["metadata"],
        "comparison": comparison_metrics(result),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

    print(f"仿真结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()

