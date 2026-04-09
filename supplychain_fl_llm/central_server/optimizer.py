from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pyomo.environ import (  # type: ignore[import-untyped]
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    Set,
    SolverFactory,
    Var,
    minimize,
    value,
)


@dataclass
class OptimizationConfig:
    emission_weight: float = 1.0
    holding_cost_rate: float = 0.12
    backup_cost_multiplier: float = 1.15
    backup_emission_multiplier: float = 1.10


def _normalize_records(forecasts: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(forecasts)
    if df.empty:
        raise ValueError("Forecast input cannot be empty")

    required = [
        "product_id",
        "forecast_demand",
        "uncertainty",
        "inventory",
        "unit_cost",
        "transport_cost",
        "emissions",
        "stockout_penalty",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Forecast input is missing fields: {missing}")

    df = df.copy()
    df["product_id"] = df["product_id"].astype(str)
    for col in required[1:]:
        df[col] = df[col].astype(float)

    return df


def _solve_with_pyomo(df: pd.DataFrame, cfg: OptimizationConfig, solver_name: str | None = None) -> dict[str, Any]:
    products = df["product_id"].tolist()
    demand = {row.product_id: float(row.forecast_demand) for row in df.itertuples()}
    uncertainty = {row.product_id: float(row.uncertainty) for row in df.itertuples()}
    inventory = {row.product_id: float(row.inventory) for row in df.itertuples()}
    unit_cost = {row.product_id: float(row.unit_cost) for row in df.itertuples()}
    transport_cost = {row.product_id: float(row.transport_cost) for row in df.itertuples()}
    emissions = {row.product_id: float(row.emissions) for row in df.itertuples()}
    stockout_penalty = {row.product_id: float(row.stockout_penalty) for row in df.itertuples()}

    safety_stock = {pid: 1.65 * uncertainty[pid] for pid in products}
    backup_floor = {pid: max(0.0, 0.15 * uncertainty[pid]) for pid in products}

    model = ConcreteModel(name="supply_chain_optimizer")
    model.P = Set(initialize=products)

    model.demand = Param(model.P, initialize=demand)
    model.safety = Param(model.P, initialize=safety_stock)
    model.inventory = Param(model.P, initialize=inventory)
    model.unit_cost = Param(model.P, initialize=unit_cost)
    model.transport_cost = Param(model.P, initialize=transport_cost)
    model.emissions = Param(model.P, initialize=emissions)
    model.stockout_penalty = Param(model.P, initialize=stockout_penalty)
    model.backup_floor = Param(model.P, initialize=backup_floor)

    model.reorder = Var(model.P, domain=NonNegativeReals)
    model.shortage = Var(model.P, domain=NonNegativeReals)
    model.primary = Var(model.P, domain=NonNegativeReals)
    model.backup = Var(model.P, domain=NonNegativeReals)

    def reorder_balance_rule(m, p):
        return m.reorder[p] == m.primary[p] + m.backup[p]

    model.reorder_balance = Constraint(model.P, rule=reorder_balance_rule)

    def demand_cover_rule(m, p):
        return m.inventory[p] + m.reorder[p] + m.shortage[p] >= m.demand[p] + m.safety[p]

    model.demand_cover = Constraint(model.P, rule=demand_cover_rule)

    def backup_minimum_rule(m, p):
        return m.backup[p] >= m.backup_floor[p]

    model.backup_minimum = Constraint(model.P, rule=backup_minimum_rule)

    def objective_rule(m):
        procurement_and_transport = sum(
            (m.unit_cost[p] + m.transport_cost[p] + cfg.holding_cost_rate * m.unit_cost[p]) * m.primary[p]
            for p in m.P
        )
        backup_cost = sum(
            cfg.backup_cost_multiplier * (m.unit_cost[p] + m.transport_cost[p]) * m.backup[p]
            for p in m.P
        )
        shortage_cost = sum(m.stockout_penalty[p] * m.shortage[p] for p in m.P)
        emission_cost = sum(
            cfg.emission_weight
            * (
                m.emissions[p] * m.primary[p]
                + cfg.backup_emission_multiplier * m.emissions[p] * m.backup[p]
            )
            for p in m.P
        )
        return procurement_and_transport + backup_cost + shortage_cost + emission_cost

    model.total_cost = Objective(rule=objective_rule, sense=minimize)

    preferred_solvers = [solver_name] if solver_name else ["appsi_highs", "highs", "glpk", "cbc"]
    chosen_solver = None
    for candidate in preferred_solvers:
        if candidate is None:
            continue
        solver = SolverFactory(candidate)
        if solver is not None and solver.available(False):
            chosen_solver = candidate
            break

    if chosen_solver is None:
        raise RuntimeError(
            "No Pyomo solver available. Install HiGHS or GLPK and retry. "
            "The caller can use heuristic fallback."
        )

    solver = SolverFactory(chosen_solver)
    result = solver.solve(model, tee=False)

    records: list[dict[str, Any]] = []
    total_cost = float(value(model.total_cost))
    total_emissions = 0.0
    weighted_service_numerator = 0.0
    weighted_service_denominator = 0.0

    for pid in products:
        reorder = float(value(model.reorder[pid]))
        shortage = float(value(model.shortage[pid]))
        primary = float(value(model.primary[pid]))
        backup = float(value(model.backup[pid]))

        demand_plus_safety = demand[pid] + safety_stock[pid]
        service = 1.0 if demand_plus_safety <= 0 else max(0.0, 1.0 - shortage / demand_plus_safety)

        emissions_total = emissions[pid] * primary + cfg.backup_emission_multiplier * emissions[pid] * backup
        total_emissions += emissions_total

        weighted_service_numerator += service * demand_plus_safety
        weighted_service_denominator += demand_plus_safety

        records.append(
            {
                "product_id": pid,
                "forecast_demand": demand[pid],
                "safety_stock": safety_stock[pid],
                "reorder_level": reorder,
                "source_primary": primary,
                "source_backup": backup,
                "projected_shortage": shortage,
                "service_level": service,
                "emissions": emissions_total,
            }
        )

    avg_service = (
        weighted_service_numerator / weighted_service_denominator
        if weighted_service_denominator > 0
        else 1.0
    )

    return {
        "status": str(result.solver.status),
        "termination_condition": str(result.solver.termination_condition),
        "solver": chosen_solver,
        "recommendations": records,
        "totals": {
            "total_cost": total_cost,
            "service_level": avg_service,
            "emissions": total_emissions,
        },
    }


def _heuristic_fallback(df: pd.DataFrame, cfg: OptimizationConfig) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total_cost = 0.0
    total_emissions = 0.0
    weighted_service_numerator = 0.0
    weighted_service_denominator = 0.0

    for row in df.itertuples():
        safety_stock = 1.65 * float(row.uncertainty)
        reorder = max(0.0, float(row.forecast_demand) + safety_stock - float(row.inventory))
        backup = min(reorder * 0.25, safety_stock)
        primary = max(0.0, reorder - backup)
        shortage = max(0.0, float(row.forecast_demand) + safety_stock - float(row.inventory) - reorder)

        service = 1.0 if (row.forecast_demand + safety_stock) <= 0 else max(
            0.0,
            1.0 - shortage / (row.forecast_demand + safety_stock),
        )

        emissions_total = row.emissions * primary + cfg.backup_emission_multiplier * row.emissions * backup
        cost = (
            (row.unit_cost + row.transport_cost + cfg.holding_cost_rate * row.unit_cost) * primary
            + cfg.backup_cost_multiplier * (row.unit_cost + row.transport_cost) * backup
            + row.stockout_penalty * shortage
            + cfg.emission_weight * emissions_total
        )

        total_cost += cost
        total_emissions += emissions_total
        weighted_service_numerator += service * (row.forecast_demand + safety_stock)
        weighted_service_denominator += row.forecast_demand + safety_stock

        records.append(
            {
                "product_id": str(row.product_id),
                "forecast_demand": float(row.forecast_demand),
                "safety_stock": float(safety_stock),
                "reorder_level": float(reorder),
                "source_primary": float(primary),
                "source_backup": float(backup),
                "projected_shortage": float(shortage),
                "service_level": float(service),
                "emissions": float(emissions_total),
            }
        )

    avg_service = (
        weighted_service_numerator / weighted_service_denominator
        if weighted_service_denominator > 0
        else 1.0
    )

    return {
        "status": "heuristic_fallback",
        "termination_condition": "solver_unavailable",
        "solver": "heuristic",
        "recommendations": records,
        "totals": {
            "total_cost": float(total_cost),
            "service_level": float(avg_service),
            "emissions": float(total_emissions),
        },
    }


def optimize_supply_chain(
    forecasts: list[dict[str, Any]],
    emission_weight: float = 1.0,
    solver_name: str | None = None,
) -> dict[str, Any]:
    """Optimize reorder and sourcing policy from aggregate forecast data."""
    cfg = OptimizationConfig(emission_weight=emission_weight)
    df = _normalize_records(forecasts)

    try:
        return _solve_with_pyomo(df=df, cfg=cfg, solver_name=solver_name)
    except Exception:
        return _heuristic_fallback(df=df, cfg=cfg)
