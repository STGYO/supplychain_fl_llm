from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Optional

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from central_server.blockchain_verifier import BlockchainVerifier
from utils.hash_utils import hash_ndarrays
from utils.logging_utils import append_jsonl, configure_logger, get_project_root, write_json


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    rounds: int = 5
    min_fit_clients: int = 2
    min_available_clients: int = 2
    local_epochs: int = 1
    blockchain_rpc_url: str = "http://127.0.0.1:8545"
    contract_artifact: str = "artifacts/contract_deployment.json"
    skip_blockchain_verification: bool = False


class BlockchainAwareFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, verifier: BlockchainVerifier, metrics_path: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self.verifier = verifier
        self.metrics_path = metrics_path
        self.logger = configure_logger("server.fl")
        self.root = get_project_root()

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}

        accepted: list[tuple[ClientProxy, FitRes]] = []
        rejected_count = 0
        rmse_values: list[float] = []
        mape_values: list[float] = []

        for client, fit_res in results:
            metrics = fit_res.metrics or {}
            node_address = str(metrics.get("node_address", ""))
            model_hash = str(metrics.get("model_hash", ""))

            is_valid = self.verifier.verify_update(node_address, server_round, model_hash)
            if is_valid:
                accepted.append((client, fit_res))
                if "val_rmse" in metrics:
                    rmse_values.append(float(metrics["val_rmse"]))
                if "val_mape" in metrics:
                    mape_values.append(float(metrics["val_mape"]))
            else:
                rejected_count += 1
                self.logger.warning(
                    "Rejected update in round %s from %s due to blockchain verification failure",
                    server_round,
                    node_address,
                )

        if not accepted:
            self.logger.error(
                "No valid updates for round %s. Keeping previous global model.",
                server_round,
            )
            round_summary = {
                "round": server_round,
                "accepted_updates": 0,
                "rejected_updates": rejected_count,
                "mean_val_rmse": None,
                "mean_val_mape": None,
                "global_model_hash": None,
            }
            append_jsonl(self.metrics_path, round_summary)
            return None, {"accepted_updates": 0.0, "rejected_updates": float(rejected_count)}

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, accepted, failures)

        global_model_hash = None
        if aggregated_parameters is not None:
            global_model_hash = hash_ndarrays(parameters_to_ndarrays(aggregated_parameters))
            write_json(
                self.root / "artifacts" / "latest_global_model.json",
                {
                    "round": server_round,
                    "global_model_hash": global_model_hash,
                },
            )

        round_summary = {
            "round": server_round,
            "accepted_updates": len(accepted),
            "rejected_updates": rejected_count,
            "mean_val_rmse": mean(rmse_values) if rmse_values else None,
            "mean_val_mape": mean(mape_values) if mape_values else None,
            "global_model_hash": global_model_hash,
        }
        append_jsonl(self.metrics_path, round_summary)

        metric_out = {
            "accepted_updates": float(len(accepted)),
            "rejected_updates": float(rejected_count),
        }
        metric_out.update(aggregated_metrics)

        self.logger.info(
            "Round %s aggregated | accepted=%s rejected=%s rmse=%.4f mape=%.4f",
            server_round,
            len(accepted),
            rejected_count,
            round_summary["mean_val_rmse"] or float("nan"),
            round_summary["mean_val_mape"] or float("nan"),
        )

        return aggregated_parameters, metric_out


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(description="Federated learning server with blockchain verification")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-fit-clients", type=int, default=2)
    parser.add_argument("--min-available-clients", type=int, default=2)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--blockchain-rpc", type=str, default="http://127.0.0.1:8545")
    parser.add_argument("--contract-artifact", type=str, default="artifacts/contract_deployment.json")
    parser.add_argument("--skip-blockchain-verification", action="store_true")
    args = parser.parse_args()

    return ServerConfig(
        host=args.host,
        port=args.port,
        rounds=args.rounds,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
        local_epochs=args.local_epochs,
        blockchain_rpc_url=args.blockchain_rpc,
        contract_artifact=args.contract_artifact,
        skip_blockchain_verification=args.skip_blockchain_verification,
    )


def build_fit_config_fn(local_epochs: int):
    def fit_config(server_round: int) -> dict[str, Scalar]:
        return {
            "server_round": str(server_round),
            "local_epochs": str(local_epochs),
        }

    return fit_config


def main() -> None:
    cfg = parse_args()
    logger = configure_logger("server.main")

    root = get_project_root()
    metrics_path = root / "artifacts" / "fl_round_metrics.jsonl"

    verifier = BlockchainVerifier(
        rpc_url=cfg.blockchain_rpc_url,
        contract_artifact_path=str((root / cfg.contract_artifact).resolve())
        if not Path(cfg.contract_artifact).is_absolute()
        else cfg.contract_artifact,
        strict=not cfg.skip_blockchain_verification,
    )

    strategy = BlockchainAwareFedAvg(
        verifier=verifier,
        metrics_path=metrics_path,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=cfg.min_fit_clients,
        min_evaluate_clients=cfg.min_fit_clients,
        min_available_clients=cfg.min_available_clients,
        on_fit_config_fn=build_fit_config_fn(cfg.local_epochs),
        accept_failures=False,
    )

    server_address = f"{cfg.host}:{cfg.port}"
    logger.info("Starting Flower server on %s for %s rounds", server_address, cfg.rounds)
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=cfg.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
