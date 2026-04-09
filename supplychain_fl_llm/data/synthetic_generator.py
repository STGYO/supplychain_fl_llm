from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd


@dataclass
class GeneratorConfig:
    partners: int = 3
    products: int = 5
    timesteps: int = 180
    seed: int = 42
    output_dir: str = "data/generated"


def _generate_partner_product_series(
    rng: np.random.Generator,
    partner_id: int,
    product_id: int,
    timesteps: int,
) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []

    base_demand = rng.uniform(35.0, 130.0)
    seasonal_amplitude = rng.uniform(0.1, 0.35) * base_demand
    seasonal_phase = rng.uniform(0.0, 2.0 * math.pi)
    trend = rng.uniform(-0.03, 0.05)

    base_lead_time = rng.uniform(2.0, 7.0)
    base_unit_cost = rng.uniform(15.0, 65.0)
    base_transport_cost = rng.uniform(2.0, 12.0)
    base_emission = rng.uniform(0.8, 5.5)

    inventory = rng.uniform(base_demand * 1.0, base_demand * 2.2)
    disruption_probability = rng.uniform(0.02, 0.07)

    for t in range(timesteps):
        seasonal_component = seasonal_amplitude * math.sin((2.0 * math.pi * t / 30.0) + seasonal_phase)
        trend_component = trend * t * base_demand / timesteps
        noise = rng.normal(0.0, base_demand * 0.08)

        disruption_flag = 1 if rng.random() < disruption_probability else 0
        disruption_demand_boost = rng.uniform(0.08, 0.30) * base_demand if disruption_flag else 0.0
        disruption_cost_boost = rng.uniform(0.05, 0.25) if disruption_flag else 0.0

        demand = max(1.0, base_demand + seasonal_component + trend_component + noise + disruption_demand_boost)

        lead_time = max(1.0, rng.normal(base_lead_time + (1.5 if disruption_flag else 0.0), 0.8))
        unit_cost = base_unit_cost * (1.0 + rng.normal(0.0, 0.05) + disruption_cost_boost)
        transport_cost = base_transport_cost * (1.0 + rng.normal(0.0, 0.08) + disruption_cost_boost)
        emissions = base_emission * (1.0 + rng.normal(0.0, 0.1) + disruption_cost_boost)
        stockout_penalty = unit_cost * rng.uniform(1.1, 2.0)

        reorder_trigger = base_demand * rng.uniform(0.7, 1.1)
        replenishment = base_demand * rng.uniform(0.6, 1.3) if inventory < reorder_trigger else 0.0

        inventory = max(0.0, inventory + replenishment - demand)

        records.append(
            {
                "partner_id": partner_id,
                "time_step": t,
                "product_id": product_id,
                "demand": round(float(demand), 4),
                "inventory": round(float(inventory), 4),
                "lead_time": round(float(lead_time), 4),
                "unit_cost": round(float(unit_cost), 4),
                "transport_cost": round(float(transport_cost), 4),
                "emissions": round(float(emissions), 4),
                "stockout_penalty": round(float(stockout_penalty), 4),
                "disruption_flag": disruption_flag,
            }
        )

    return records


def generate_synthetic_data(config: GeneratorConfig) -> list[Path]:
    rng = np.random.default_rng(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    produced_files: list[Path] = []
    combined_rows: list[dict[str, float | int]] = []

    for partner_id in range(1, config.partners + 1):
        partner_rows: list[dict[str, float | int]] = []
        for product_id in range(1, config.products + 1):
            series = _generate_partner_product_series(rng, partner_id, product_id, config.timesteps)
            partner_rows.extend(series)

        partner_df = pd.DataFrame(partner_rows)
        partner_df.sort_values(["product_id", "time_step"], inplace=True)

        partner_path = output_dir / f"partner_{partner_id}.csv"
        partner_df.to_csv(partner_path, index=False)
        produced_files.append(partner_path)

        combined_rows.extend(partner_rows)

    combined_df = pd.DataFrame(combined_rows)
    combined_df.sort_values(["partner_id", "product_id", "time_step"], inplace=True)
    combined_path = output_dir / "all_partners.csv"
    combined_df.to_csv(combined_path, index=False)
    produced_files.append(combined_path)

    return produced_files


def parse_args() -> GeneratorConfig:
    parser = argparse.ArgumentParser(description="Synthetic supply chain dataset generator")
    parser.add_argument("--partners", type=int, default=3)
    parser.add_argument("--products", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/generated")
    args = parser.parse_args()

    return GeneratorConfig(
        partners=args.partners,
        products=args.products,
        timesteps=args.timesteps,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main() -> None:
    config = parse_args()
    files = generate_synthetic_data(config)
    print("Synthetic data generated:")
    for file_path in files:
        print(f"- {file_path}")


if __name__ == "__main__":
    main()
