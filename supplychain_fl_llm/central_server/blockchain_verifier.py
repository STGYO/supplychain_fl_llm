from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from web3 import Web3

from utils.logging_utils import append_jsonl, configure_logger, get_project_root


class BlockchainVerifier:
    def __init__(
        self,
        rpc_url: str,
        contract_artifact_path: str,
        strict: bool = True,
    ) -> None:
        self.logger = configure_logger("server.blockchain_verifier")
        self.strict = strict

        self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
        if not self.w3.is_connected():
            if strict:
                raise ConnectionError(f"Cannot connect to blockchain RPC: {rpc_url}")
            self.logger.warning("Blockchain unreachable, running verifier in permissive mode")
            self.contract = None
            return

        artifact_path = Path(contract_artifact_path)
        if not artifact_path.exists():
            if strict:
                raise FileNotFoundError(f"Contract artifact not found: {artifact_path}")
            self.logger.warning("Contract artifact not found, running verifier in permissive mode")
            self.contract = None
            return

        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(artifact["contract_address"]),
            abi=artifact["abi"],
        )

        root = get_project_root()
        self.audit_log = root / "logs" / "blockchain_verification.jsonl"

    def fetch_round_updates(self, round_number: int) -> list[dict[str, Any]]:
        if self.contract is None:
            return []

        rows = self.contract.functions.getUpdates(int(round_number)).call()
        updates: list[dict[str, Any]] = []
        for row in rows:
            updates.append(
                {
                    "node_address": row[0],
                    "round_number": int(row[1]),
                    "model_hash": row[2],
                    "timestamp": int(row[3]),
                }
            )
        return updates

    def verify_update(self, node_address: str, round_number: int, model_hash: str) -> bool:
        if self.contract is None:
            return not self.strict

        try:
            normalized = Web3.to_checksum_address(node_address)
        except Exception:  # noqa: BLE001 - malformed address should fail verification.
            normalized = node_address

        updates = self.fetch_round_updates(round_number)
        is_valid = any(
            Web3.to_checksum_address(update["node_address"]) == normalized
            and update["model_hash"] == model_hash
            for update in updates
        )

        append_jsonl(
            self.audit_log,
            {
                "node_address": node_address,
                "round_number": round_number,
                "model_hash": model_hash,
                "is_valid": is_valid,
            },
        )

        return is_valid

    def summarize_round(self, round_number: int) -> dict[str, Any]:
        updates = self.fetch_round_updates(round_number)
        node_count = len({row["node_address"] for row in updates})
        return {
            "round_number": round_number,
            "update_count": len(updates),
            "unique_nodes": node_count,
        }
