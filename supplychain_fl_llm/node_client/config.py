from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import socket


@dataclass
class NodeConfig:
    node_id: str
    partner_id: int
    server_address: str
    blockchain_rpc_url: str
    contract_artifact: str
    dataset_path: str
    account_index: int
    account_address: str | None
    private_key: str | None
    local_epochs: int
    batch_size: int
    sequence_length: int
    hidden_size: int
    learning_rate: float
    max_retries: int


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_node_id() -> str:
    return socket.gethostname()


def build_parser() -> argparse.ArgumentParser:
    project_root = _default_project_root()

    parser = argparse.ArgumentParser(description="Federated client node for supply chain FL")
    parser.add_argument("--node-id", type=str, default=os.getenv("NODE_ID", _default_node_id()))
    parser.add_argument("--partner-id", type=int, default=int(os.getenv("PARTNER_ID", "1")))
    parser.add_argument("--server", type=str, default=os.getenv("FL_SERVER", "127.0.0.1:8080"))
    parser.add_argument(
        "--blockchain-rpc",
        type=str,
        default=os.getenv("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545"),
    )
    parser.add_argument(
        "--contract-artifact",
        type=str,
        default=os.getenv(
            "CONTRACT_ARTIFACT",
            str(project_root / "artifacts" / "contract_deployment.json"),
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to local partner CSV. Defaults to data/generated/partner_<partner-id>.csv",
    )
    parser.add_argument("--account-index", type=int, default=int(os.getenv("ACCOUNT_INDEX", "0")))
    parser.add_argument("--account-address", type=str, default=os.getenv("ACCOUNT_ADDRESS", ""))
    parser.add_argument("--private-key", type=str, default=os.getenv("PRIVATE_KEY", ""))
    parser.add_argument("--local-epochs", type=int, default=int(os.getenv("LOCAL_EPOCHS", "1")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "32")))
    parser.add_argument("--sequence-length", type=int, default=int(os.getenv("SEQUENCE_LENGTH", "12")))
    parser.add_argument("--hidden-size", type=int, default=int(os.getenv("HIDDEN_SIZE", "64")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.001")))
    parser.add_argument("--max-retries", type=int, default=int(os.getenv("MAX_RETRIES", "5")))
    return parser


def parse_node_config(argv: list[str] | None = None) -> NodeConfig:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = _default_project_root()
    dataset_path = args.dataset or str(project_root / "data" / "generated" / f"partner_{args.partner_id}.csv")

    account_address = args.account_address.strip() or None
    private_key = args.private_key.strip() or None

    return NodeConfig(
        node_id=args.node_id,
        partner_id=args.partner_id,
        server_address=args.server,
        blockchain_rpc_url=args.blockchain_rpc,
        contract_artifact=args.contract_artifact,
        dataset_path=dataset_path,
        account_index=args.account_index,
        account_address=account_address,
        private_key=private_key,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        max_retries=args.max_retries,
    )
