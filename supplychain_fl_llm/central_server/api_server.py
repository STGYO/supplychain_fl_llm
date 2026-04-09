from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from central_server.llm_engine import build_llm_engine_from_env
from central_server.optimizer import optimize_supply_chain
from utils.logging_utils import append_jsonl, read_json, read_jsonl, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"

FORECAST_FILE = ARTIFACTS_DIR / "latest_forecasts.json"
OPTIMIZATION_FILE = ARTIFACTS_DIR / "latest_optimization.json"
LLM_FILE = ARTIFACTS_DIR / "latest_llm.json"
FL_METRICS_FILE = ARTIFACTS_DIR / "fl_round_metrics.jsonl"
FEEDBACK_FILE = LOGS_DIR / "feedback.jsonl"


class ForecastItem(BaseModel):
    product_id: str | int
    forecast_demand: float = Field(gt=0)
    uncertainty: float = Field(ge=0)
    inventory: float = Field(ge=0)
    unit_cost: float = Field(gt=0)
    transport_cost: float = Field(ge=0)
    emissions: float = Field(ge=0)
    stockout_penalty: float = Field(gt=0)


class ForecastPayload(BaseModel):
    forecasts: list[ForecastItem]
    emission_weight: float = Field(default=1.0, ge=0)


class LLMRequest(BaseModel):
    manager_question: str = Field(
        default="Explain why inventory should increase at the distributor node given forecast uncertainty and emission constraints."
    )


class FeedbackPayload(BaseModel):
    user_id: str
    action: Literal["accept", "modify", "reject"]
    recommendation_text: str
    modified_text: str | None = None
    decision_time_seconds: float = Field(gt=0)
    comment: str | None = None


app = FastAPI(title="Supply Chain FL + Blockchain + LLM API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_engine = build_llm_engine_from_env()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Supply Chain Analytics API is running"}


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "artifacts": {
            "forecasts": FORECAST_FILE.exists(),
            "optimization": OPTIMIZATION_FILE.exists(),
            "llm": LLM_FILE.exists(),
            "fl_metrics": FL_METRICS_FILE.exists(),
        },
    }


@app.get("/metrics/fl")
def get_fl_metrics(limit: int = 100) -> list[dict]:
    return read_jsonl(FL_METRICS_FILE, limit=limit)


@app.get("/forecasts/latest")
def get_latest_forecasts() -> dict:
    payload = read_json(FORECAST_FILE)
    if payload is None:
        raise HTTPException(status_code=404, detail="No forecast artifact available")
    return payload


@app.post("/forecasts/latest")
def set_latest_forecasts(payload: ForecastPayload) -> dict:
    output = {
        "forecasts": [item.model_dump() for item in payload.forecasts],
        "emission_weight": payload.emission_weight,
    }
    write_json(FORECAST_FILE, output)
    return output


@app.post("/optimization/run")
def run_optimization(payload: ForecastPayload | None = Body(default=None)) -> dict:
    if payload is None:
        stored = read_json(FORECAST_FILE)
        if stored is None:
            raise HTTPException(status_code=400, detail="No forecast payload provided and no stored forecasts found")

        forecasts = stored.get("forecasts", [])
        emission_weight = float(stored.get("emission_weight", 1.0))
    else:
        forecasts = [item.model_dump() for item in payload.forecasts]
        emission_weight = payload.emission_weight
        write_json(
            FORECAST_FILE,
            {
                "forecasts": forecasts,
                "emission_weight": emission_weight,
            },
        )

    result = optimize_supply_chain(forecasts=forecasts, emission_weight=emission_weight)
    write_json(OPTIMIZATION_FILE, result)
    return result


@app.get("/optimization/latest")
def get_latest_optimization() -> dict:
    payload = read_json(OPTIMIZATION_FILE)
    if payload is None:
        raise HTTPException(status_code=404, detail="No optimization result available")
    return payload


@app.post("/llm/explain")
def run_llm_explanation(request: LLMRequest) -> dict:
    optimization = read_json(OPTIMIZATION_FILE)
    if optimization is None:
        raise HTTPException(status_code=400, detail="Run optimization before requesting LLM explanation")

    result = llm_engine.generate_recommendations(
        optimization_output=optimization,
        manager_question=request.manager_question,
    )
    write_json(LLM_FILE, result)
    return result


@app.get("/llm/latest")
def get_latest_llm() -> dict:
    payload = read_json(LLM_FILE)
    if payload is None:
        raise HTTPException(status_code=404, detail="No LLM explanation available")
    return payload


@app.post("/feedback")
def submit_feedback(payload: FeedbackPayload) -> dict:
    row = payload.model_dump()
    append_jsonl(FEEDBACK_FILE, row)
    return {"status": "logged", "feedback": row}


@app.get("/feedback")
def list_feedback(limit: int = 200) -> list[dict]:
    return read_jsonl(FEEDBACK_FILE, limit=limit)


@app.get("/feedback/summary")
def feedback_summary() -> dict:
    rows = read_jsonl(FEEDBACK_FILE)
    if not rows:
        return {
            "count": 0,
            "acceptance_rate": 0.0,
            "avg_decision_time_seconds": 0.0,
        }

    accept_count = sum(1 for row in rows if row.get("action") == "accept")
    avg_decision_time = sum(float(row.get("decision_time_seconds", 0.0)) for row in rows) / len(rows)

    return {
        "count": len(rows),
        "acceptance_rate": accept_count / len(rows),
        "avg_decision_time_seconds": avg_decision_time,
    }


def main() -> None:
    uvicorn.run("central_server.api_server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
