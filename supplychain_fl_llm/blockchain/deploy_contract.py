from __future__ import annotations

import argparse
from pathlib import Path
import json

from solcx import compile_standard, install_solc
from web3 import Web3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy FederatedUpdateRegistry contract to Ganache")
    parser.add_argument("--rpc-url", type=str, default="http://127.0.0.1:8545")
    parser.add_argument("--solc-version", type=str, default="0.8.20")
    parser.add_argument("--contract-file", type=str, default="blockchain/contract.sol")
    parser.add_argument("--output", type=str, default="artifacts/contract_deployment.json")
    parser.add_argument("--account-index", type=int, default=0)
    parser.add_argument("--private-key", type=str, default="")
    return parser.parse_args()


def _compile_contract(contract_path: Path, solc_version: str) -> tuple[list[dict], str]:
    source = contract_path.read_text(encoding="utf-8")
    install_solc(solc_version)

    compiled = compile_standard(
        {
            "language": "Solidity",
            "sources": {contract_path.name: {"content": source}},
            "settings": {
                "optimizer": {"enabled": True, "runs": 200},
                "outputSelection": {
                    "*": {
                        "*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]
                    }
                },
            },
        },
        solc_version=solc_version,
    )

    contract_data = compiled["contracts"][contract_path.name]["FederatedUpdateRegistry"]
    abi = contract_data["abi"]
    bytecode = contract_data["evm"]["bytecode"]["object"]

    if not bytecode:
        raise RuntimeError("Compilation failed to produce bytecode")

    return abi, bytecode


def _deploy(
    w3: Web3,
    abi: list[dict],
    bytecode: str,
    account_index: int,
    private_key: str | None,
) -> tuple[str, str]:
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    if private_key:
        account = w3.eth.account.from_key(private_key)
        tx = contract.constructor().build_transaction(
            {
                "from": account.address,
                "nonce": w3.eth.get_transaction_count(account.address),
                "gas": 3_500_000,
                "gasPrice": w3.to_wei("2", "gwei"),
                "chainId": w3.eth.chain_id,
            }
        )
        signed = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    else:
        accounts = w3.eth.accounts
        if not accounts:
            raise RuntimeError("No unlocked accounts available from RPC provider")
        if account_index >= len(accounts):
            raise IndexError(f"account-index {account_index} is out of range")

        sender = accounts[account_index]
        tx_hash = contract.constructor().transact(
            {
                "from": sender,
                "gas": 3_500_000,
            }
        )

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    if int(receipt.status) != 1:
        raise RuntimeError("Contract deployment failed")

    return receipt.contractAddress, tx_hash.hex()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    contract_path = (project_root / args.contract_file).resolve()
    output_path = (project_root / args.output).resolve()

    if not contract_path.exists():
        raise FileNotFoundError(f"Contract file not found: {contract_path}")

    print(f"Compiling contract: {contract_path}")
    abi, bytecode = _compile_contract(contract_path, args.solc_version)

    print(f"Connecting to RPC: {args.rpc_url}")
    w3 = Web3(Web3.HTTPProvider(args.rpc_url, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        raise ConnectionError(f"Could not connect to blockchain RPC at {args.rpc_url}")

    print("Deploying FederatedUpdateRegistry...")
    contract_address, tx_hash = _deploy(
        w3=w3,
        abi=abi,
        bytecode=bytecode,
        account_index=args.account_index,
        private_key=args.private_key.strip() or None,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rpc_url": args.rpc_url,
        "contract_name": "FederatedUpdateRegistry",
        "contract_address": contract_address,
        "deployment_tx_hash": tx_hash,
        "chain_id": int(w3.eth.chain_id),
        "abi": abi,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Deployment successful")
    print(f"- Address: {contract_address}")
    print(f"- Tx hash: {tx_hash}")
    print(f"- Artifact: {output_path}")


if __name__ == "__main__":
    main()
