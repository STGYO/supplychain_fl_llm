# Ganache Setup (Private Ethereum for FL Update Logging)

This guide configures Ganache so client laptops can submit model-hash transactions directly.

## 1) Install Ganache CLI

```powershell
npm install -g ganache
```

If your environment still uses `ganache-cli`, you can replace `ganache` with `ganache-cli` in the commands below.

## 2) Start Ganache on the Server Laptop

Run this from the server laptop (example server LAN IP: `192.168.1.10`):

```powershell
ganache --host 0.0.0.0 --port 8545 --chain.chainId 1337 --wallet.totalAccounts 20 --wallet.defaultBalance 500
```

Notes:
- `--host 0.0.0.0` exposes RPC over LAN for direct client submissions.
- Ensure Windows firewall allows inbound TCP on port `8545`.

## 3) Verify RPC Reachability

On a client laptop:

```powershell
curl http://192.168.1.10:8545
```

A JSON-RPC response confirms network reachability.

## 4) Deploy Smart Contract

From the project root on the server laptop:

```powershell
python blockchain/deploy_contract.py --rpc-url http://127.0.0.1:8545 --account-index 0
```

This writes the deployment artifact to:

`artifacts/contract_deployment.json`

## 5) Configure Clients

Client nodes should point to:
- `--blockchain-rpc http://192.168.1.10:8545`
- `--contract-artifact` copied from server deployment artifact

In demos, you can copy the artifact JSON to each client machine and keep the same contract address.
