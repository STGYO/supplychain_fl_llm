from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from web3 import Web3


class BlockchainClient:
    def __init__(
        self,
        rpc_url: str,
        contract_artifact_path: str,
        account_index: int = 0,
        account_address: str | None = None,
        private_key: str | None = None,
        request_timeout: int = 30,
    ) -> None:
        self.rpc_url = rpc_url
        self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": request_timeout}))
        if not self.w3.is_connected():
            raise ConnectionError(f"Unable to connect to blockchain RPC: {rpc_url}")

        artifact = self._load_artifact(contract_artifact_path)
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(artifact["contract_address"]),
            abi=artifact["abi"],
        )

        self.private_key = private_key
        self.chain_id = int(artifact.get("chain_id") or self.w3.eth.chain_id)
        self.account_address = self._resolve_account(account_index, account_address, private_key)

    @staticmethod
    def _load_artifact(path: str) -> dict[str, Any]:
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Contract artifact not found: {artifact_path}")

        with artifact_path.open("r", encoding="utf-8") as fp:
            artifact = json.load(fp)

        if "contract_address" not in artifact or "abi" not in artifact:
            raise ValueError("Contract artifact must include contract_address and abi")

        return artifact

    def _resolve_account(self, account_index: int, account_address: str | None, private_key: str | None) -> str:
        if account_address:
            return Web3.to_checksum_address(account_address)

        if private_key:
            account = self.w3.eth.account.from_key(private_key)
            return Web3.to_checksum_address(account.address)

        accounts = self.w3.eth.accounts
        if not accounts:
            raise ValueError(
                "No unlocked accounts found in RPC provider. Provide --account-address and --private-key."
            )

        if account_index >= len(accounts):
            raise IndexError(f"Account index {account_index} out of range for available accounts")

        return Web3.to_checksum_address(accounts[account_index])

    def _send_transaction(self, fn: Any) -> str:
        if self.private_key:
            tx = fn.build_transaction(
                {
                    "from": self.account_address,
                    "nonce": self.w3.eth.get_transaction_count(self.account_address),
                    "chainId": self.chain_id,
                    "gas": 500000,
                    "gasPrice": self.w3.to_wei("2", "gwei"),
                }
            )
            signed = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        else:
            tx_hash = fn.transact(
                {
                    "from": self.account_address,
                    "gas": 500000,
                }
            )

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if int(receipt.status) != 1:
            raise RuntimeError("Blockchain transaction failed")

        return tx_hash.hex()

    def register_node(self) -> str | None:
        already_registered = self.contract.functions.registeredNodes(self.account_address).call()
        if already_registered:
            return None

        return self._send_transaction(self.contract.functions.registerNode())

    def submit_update(self, round_number: int, model_hash: str) -> str:
        fn = self.contract.functions.submitUpdate(int(round_number), model_hash)
        return self._send_transaction(fn)

    def get_updates(self, round_number: int) -> list[dict[str, Any]]:
        rows = self.contract.functions.getUpdates(int(round_number)).call()
        output: list[dict[str, Any]] = []
        for row in rows:
            output.append(
                {
                    "node_address": row[0],
                    "round_number": int(row[1]),
                    "model_hash": row[2],
                    "timestamp": int(row[3]),
                }
            )
        return output

    def has_update(self, node_address: str, round_number: int, model_hash: str) -> bool:
        normalized = Web3.to_checksum_address(node_address)
        updates = self.get_updates(round_number)
        for update in updates:
            if (
                Web3.to_checksum_address(update["node_address"]) == normalized
                and update["model_hash"] == model_hash
            ):
                return True
        return False
