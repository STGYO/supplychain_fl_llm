# Federated Learning and LLMs for Privacy-Preserving Human-Centric Supply Chain Analytics

This repository is a complete research prototype for a multi-laptop supply chain analytics system using:
- Federated learning (Flower + PyTorch LSTM)
- Private blockchain logging and verification (Ganache + Solidity + Web3.py)
- Optimization policy generation (Pyomo)
- LLM-based managerial reasoning (LangChain + LM Studio-compatible endpoint)
- Human-in-the-loop dashboard (Streamlit)

Raw operational data remains local to each client node. Only model updates and metadata are shared.

## Repository Layout

```text
supplychain_fl_llm/
  central_server/
    fl_server.py
    optimizer.py
    llm_engine.py
    blockchain_verifier.py
    api_server.py
  node_client/
    node.py
    trainer.py
    blockchain_client.py
    dataset_loader.py
    config.py
  blockchain/
    contract.sol
    deploy_contract.py
    ganache_setup.md
  data/
    synthetic_generator.py
  dashboard/
    streamlit_app.py
  experiments/
    run_experiment.py
  utils/
    hash_utils.py
    logging_utils.py
    network_utils.py
  requirements.txt
  README.md
```

## System Topology

- Server laptop:
  - Flower aggregation server on `0.0.0.0:8080`
  - Blockchain verifier and Ganache node
  - Optimization engine
  - LLM decision layer
  - FastAPI service
  - Streamlit dashboard
- Client laptops (3 recommended for demo):
  - Local dataset and local training
  - Model update hashing
  - Direct blockchain transaction submission
  - FL client update exchange with server

Example LAN deployment:
- Server: `192.168.1.10`
- Clients: `192.168.1.11`, `192.168.1.12`, `192.168.1.13`

## Prerequisites

- Python `3.11`
- Node.js + npm (for Ganache)
- Optional but recommended for Pyomo exact solving:
  - HiGHS or GLPK solver
- Optional for LLM reasoning:
  - LM Studio running OpenAI-compatible API at `http://127.0.0.1:1234/v1`

## Setup

### 1) Create environment and install dependencies

From project root (`supplychain_fl_llm`):

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Generate synthetic datasets

```powershell
python data/synthetic_generator.py --partners 3 --products 5 --timesteps 180 --output-dir data/generated
```

This creates per-partner datasets and a combined dataset.

## Demo Workflow (Multi-Laptop)

### Step 1: Start blockchain (server laptop)

```powershell
ganache --host 0.0.0.0 --port 8545 --chain.chainId 1337 --wallet.totalAccounts 20 --wallet.defaultBalance 500
```

### Step 2: Deploy smart contract (server laptop)

```powershell
python blockchain/deploy_contract.py --rpc-url http://127.0.0.1:8545 --account-index 0
```

Artifact output:
- `artifacts/contract_deployment.json`

Copy this artifact file to each client laptop at the same relative path.

### Step 3: Start federated server (server laptop)

```powershell
python central_server/fl_server.py --host 0.0.0.0 --port 8080 --rounds 5 --blockchain-rpc http://127.0.0.1:8545 --contract-artifact artifacts/contract_deployment.json
```

### Step 4: Run clients on other laptops

On each client laptop, use a distinct partner and account index:

```powershell
python node_client/node.py --server 192.168.1.10:8080 --partner-id 1 --blockchain-rpc http://192.168.1.10:8545 --contract-artifact artifacts/contract_deployment.json --account-index 1
```

```powershell
python node_client/node.py --server 192.168.1.10:8080 --partner-id 2 --blockchain-rpc http://192.168.1.10:8545 --contract-artifact artifacts/contract_deployment.json --account-index 2
```

```powershell
python node_client/node.py --server 192.168.1.10:8080 --partner-id 3 --blockchain-rpc http://192.168.1.10:8545 --contract-artifact artifacts/contract_deployment.json --account-index 3
```

### Step 5: Start API and dashboard (server laptop)

```powershell
python central_server/api_server.py
```

```powershell
streamlit run dashboard/streamlit_app.py
```

## Optional: Run Scenario Evaluation Suite

This script runs:
1. Local model only
2. Centralized baseline
3. Federated learning
4. Federated + optimization + LLM

```powershell
python experiments/run_experiment.py --partners 3 --products 5 --timesteps 180 --rounds 5
```

Output artifacts include:
- `artifacts/experiment_results.csv`
- `artifacts/experiment_results.json`
- `artifacts/latest_forecasts.json`
- `artifacts/latest_optimization.json`
- `artifacts/latest_llm.json`

## Metrics Captured

### Forecast accuracy
- RMSE
- MAPE

### Supply chain performance
- Total cost
- Service level
- Emissions

### Human decision metrics
- Decision time
- Acceptance rate

## Privacy and Blockchain Guarantees

- Raw partner operational rows never leave client laptops.
- Each client computes a deterministic hash of local model weights.
- Hashes are submitted on-chain per training round.
- Aggregation server verifies hash presence before accepting updates.
- Immutable logs provide auditable training participation records.

## API Endpoints

- `GET /health`
- `GET /metrics/fl`
- `GET /forecasts/latest`
- `POST /forecasts/latest`
- `POST /optimization/run`
- `GET /optimization/latest`
- `POST /llm/explain`
- `GET /llm/latest`
- `POST /feedback`
- `GET /feedback`
- `GET /feedback/summary`

## Troubleshooting

- Clients cannot reach server:
  - Check firewall rules for ports `8080` and `8545`.
  - Confirm server binds to `0.0.0.0`.
- Blockchain verification fails:
  - Verify clients use same contract artifact and correct chain RPC.
  - Ensure each client uses a unique Ganache account.
- Pyomo solver unavailable:
  - Install HiGHS or GLPK for exact solving.
  - Code falls back to a deterministic heuristic if solver is missing.
- LLM unavailable:
  - Start LM Studio API endpoint.
  - System will use deterministic fallback explanations if endpoint is down.

## Academic Use Notes

This prototype is suitable for:
- Graduate coursework demonstrations
- Federated analytics research baselines
- Conference demos requiring explainable human-in-the-loop decisions

For production deployment, add enterprise security hardening, robust key management, and distributed validator fault tolerance.
