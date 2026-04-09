from __future__ import annotations

import time

import flwr as fl

from node_client.blockchain_client import BlockchainClient
from node_client.config import parse_node_config
from node_client.dataset_loader import ensure_partner_dataset, load_sequence_dataset
from node_client.trainer import SupplyChainTrainer, TrainerConfig
from utils.hash_utils import hash_ndarrays
from utils.logging_utils import append_jsonl, configure_logger, get_project_root


class SupplyChainNumPyClient(fl.client.NumPyClient):
    def __init__(
        self,
        node_id: str,
        trainer: SupplyChainTrainer,
        dataset,
        blockchain_client: BlockchainClient,
        default_local_epochs: int,
    ) -> None:
        self.node_id = node_id
        self.trainer = trainer
        self.dataset = dataset
        self.blockchain_client = blockchain_client
        self.default_local_epochs = default_local_epochs
        self.logger = configure_logger(f"node.{node_id}")

        root = get_project_root()
        self.training_log_path = root / "logs" / f"node_{node_id}_training.jsonl"

    def get_parameters(self, config):
        return self.trainer.get_weights()

    def fit(self, parameters, config):
        server_round = int(config.get("server_round", 0))
        local_epochs = int(config.get("local_epochs", self.default_local_epochs))

        self.trainer.set_weights(parameters)

        train_metrics = self.trainer.train(
            self.dataset.train_x,
            self.dataset.train_y,
            epochs=local_epochs,
        )
        val_metrics = self.trainer.evaluate(self.dataset.val_x, self.dataset.val_y)

        updated_weights = self.trainer.get_weights()
        model_hash = hash_ndarrays(updated_weights)
        tx_hash = self.blockchain_client.submit_update(server_round, model_hash)

        metrics = {
            "node_id": self.node_id,
            "node_address": self.blockchain_client.account_address,
            "model_hash": model_hash,
            "tx_hash": tx_hash,
            "round": server_round,
            "train_loss": float(train_metrics["train_loss"]),
            "val_rmse": float(val_metrics["rmse"]),
            "val_mape": float(val_metrics["mape"]),
        }

        append_jsonl(self.training_log_path, metrics)
        self.logger.info(
            "Round=%s complete | hash=%s | rmse=%.4f | mape=%.2f",
            server_round,
            model_hash[:10],
            metrics["val_rmse"],
            metrics["val_mape"],
        )

        return updated_weights, len(self.dataset.train_x), metrics

    def evaluate(self, parameters, config):
        self.trainer.set_weights(parameters)
        scores = self.trainer.evaluate(self.dataset.val_x, self.dataset.val_y)
        return float(scores["loss"]), len(self.dataset.val_x), {
            "rmse": float(scores["rmse"]),
            "mape": float(scores["mape"]),
        }


def run() -> None:
    cfg = parse_node_config()
    logger = configure_logger(f"node.main.{cfg.node_id}")

    dataset_path = ensure_partner_dataset(cfg.dataset_path, cfg.partner_id)
    dataset = load_sequence_dataset(str(dataset_path), sequence_length=cfg.sequence_length)

    trainer = SupplyChainTrainer(
        input_size=dataset.train_x.shape[2],
        config=TrainerConfig(
            learning_rate=cfg.learning_rate,
            hidden_size=cfg.hidden_size,
            batch_size=cfg.batch_size,
        ),
    )

    blockchain_client = BlockchainClient(
        rpc_url=cfg.blockchain_rpc_url,
        contract_artifact_path=cfg.contract_artifact,
        account_index=cfg.account_index,
        account_address=cfg.account_address,
        private_key=cfg.private_key,
    )

    register_tx = blockchain_client.register_node()
    if register_tx:
        logger.info("Node registered on chain: %s", register_tx)
    else:
        logger.info("Node already registered on chain: %s", blockchain_client.account_address)

    client = SupplyChainNumPyClient(
        node_id=cfg.node_id,
        trainer=trainer,
        dataset=dataset,
        blockchain_client=blockchain_client,
        default_local_epochs=cfg.local_epochs,
    )

    retries = 0
    while True:
        try:
            logger.info("Connecting to FL server: %s", cfg.server_address)
            fl.client.start_numpy_client(server_address=cfg.server_address, client=client)
            break
        except Exception as exc:  # noqa: BLE001 - keep retry loop resilient in demos.
            retries += 1
            logger.error("Client connection or training error (attempt %s): %s", retries, exc)
            if retries >= cfg.max_retries:
                raise
            time.sleep(3)


if __name__ == "__main__":
    run()
