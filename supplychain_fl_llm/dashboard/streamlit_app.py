from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


DEFAULT_API_URL = "http://127.0.0.1:8000"


def _safe_get(api_base: str, path: str):
    try:
        response = requests.get(f"{api_base}{path}", timeout=8)
        if response.status_code >= 400:
            return None
        return response.json()
    except Exception:
        return None


def _safe_post(api_base: str, path: str, payload: dict):
    try:
        response = requests.post(f"{api_base}{path}", json=payload, timeout=15)
        if response.status_code >= 400:
            return None
        return response.json()
    except Exception:
        return None


def _load_local_artifact(filename: str):
    root = Path(__file__).resolve().parents[1]
    file_path = root / "artifacts" / filename
    if not file_path.exists():
        return None
    return pd.read_json(file_path)


def render_fl_metrics(api_base: str) -> None:
    st.subheader("Federated Learning Metrics")
    fl_rows = _safe_get(api_base, "/metrics/fl")
    if not fl_rows:
        st.info("No FL metric rows found yet. Start server/clients or run experiments.")
        return

    df = pd.DataFrame(fl_rows)
    if "round" in df.columns and "mean_val_rmse" in df.columns:
        fig = px.line(df, x="round", y="mean_val_rmse", title="Round RMSE")
        st.plotly_chart(fig, use_container_width=True)
    if "round" in df.columns and "mean_val_mape" in df.columns:
        fig = px.line(df, x="round", y="mean_val_mape", title="Round MAPE")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df, use_container_width=True)


def render_forecasts(api_base: str) -> dict | None:
    st.subheader("Forecasts")
    forecasts_payload = _safe_get(api_base, "/forecasts/latest")
    if not forecasts_payload:
        st.warning("No forecast payload available from API.")
        return None

    forecasts = forecasts_payload.get("forecasts", [])
    if not forecasts:
        st.warning("Forecast payload is empty.")
        return forecasts_payload

    df = pd.DataFrame(forecasts)
    st.dataframe(df, use_container_width=True)

    if "product_id" in df.columns and "forecast_demand" in df.columns:
        fig = px.bar(df, x="product_id", y="forecast_demand", title="Forecast Demand by Product")
        st.plotly_chart(fig, use_container_width=True)

    return forecasts_payload


def render_optimization(api_base: str, forecasts_payload: dict | None) -> dict | None:
    st.subheader("Optimization Results")
    c1, c2 = st.columns(2)

    with c1:
        run_opt = st.button("Run Optimization", use_container_width=True)
    with c2:
        refresh_opt = st.button("Refresh Optimization", use_container_width=True)

    if run_opt:
        payload = forecasts_payload
        if payload is None:
            st.error("Cannot run optimization because forecasts are unavailable.")
        else:
            _safe_post(api_base, "/optimization/run", payload)
            time.sleep(0.2)

    if refresh_opt or run_opt:
        time.sleep(0.1)

    optimization = _safe_get(api_base, "/optimization/latest")
    if not optimization:
        st.warning("No optimization output available yet.")
        return None

    totals = optimization.get("totals", {})
    r1, r2, r3 = st.columns(3)
    r1.metric("Total Cost", f"{totals.get('total_cost', 0.0):,.2f}")
    r2.metric("Service Level", f"{totals.get('service_level', 0.0):.3f}")
    r3.metric("Emissions", f"{totals.get('emissions', 0.0):,.2f}")

    recs = optimization.get("recommendations", [])
    if recs:
        rec_df = pd.DataFrame(recs)
        st.dataframe(rec_df, use_container_width=True)
        if "product_id" in rec_df.columns and "reorder_level" in rec_df.columns:
            fig = px.bar(rec_df, x="product_id", y="reorder_level", title="Reorder Levels")
            st.plotly_chart(fig, use_container_width=True)

    return optimization


def render_llm(api_base: str, optimization: dict | None) -> dict | None:
    st.subheader("LLM Decision Intelligence")

    manager_question = st.text_area(
        "Manager Question",
        value=(
            "Explain why inventory should increase at the distributor node "
            "given forecast uncertainty and emission constraints."
        ),
        height=100,
    )

    run_llm = st.button("Generate LLM Explanation", use_container_width=True)
    if run_llm:
        if optimization is None:
            st.error("Run optimization before requesting an explanation.")
        else:
            _safe_post(api_base, "/llm/explain", {"manager_question": manager_question})
            time.sleep(0.2)

    llm_payload = _safe_get(api_base, "/llm/latest")
    if not llm_payload:
        st.info("No LLM output available yet.")
        return None

    st.caption(f"Mode: {llm_payload.get('mode', 'unknown')}")
    st.write(llm_payload.get("explanation", ""))

    ranked = llm_payload.get("ranked_recommendations", [])
    if ranked:
        st.write("Top Ranked Recommendations")
        st.dataframe(pd.DataFrame(ranked), use_container_width=True)

    return llm_payload


def render_feedback(api_base: str, llm_payload: dict | None) -> None:
    st.subheader("Human-in-the-Loop Feedback")

    default_recommendation = ""
    if llm_payload and llm_payload.get("ranked_recommendations"):
        top = llm_payload["ranked_recommendations"][0]
        default_recommendation = (
            f"Product {top.get('product_id')} reorder {top.get('reorder_level', 0.0):.2f}"
        )

    with st.form("feedback_form"):
        user_id = st.text_input("User ID", value="manager_1")
        action = st.selectbox("Action", ["accept", "modify", "reject"])
        recommendation_text = st.text_area("Recommendation", value=default_recommendation, height=80)
        modified_text = st.text_area("Modified Recommendation (optional)", value="", height=80)
        decision_time = st.number_input("Decision Time (seconds)", min_value=0.1, value=12.0, step=0.5)
        comment = st.text_area("Comment", value="", height=80)

        submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        payload = {
            "user_id": user_id,
            "action": action,
            "recommendation_text": recommendation_text,
            "modified_text": modified_text or None,
            "decision_time_seconds": decision_time,
            "comment": comment or None,
        }
        result = _safe_post(api_base, "/feedback", payload)
        if result:
            st.success("Feedback logged")
        else:
            st.error("Failed to log feedback")

    summary = _safe_get(api_base, "/feedback/summary")
    if summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("Feedback Count", f"{summary.get('count', 0)}")
        c2.metric("Acceptance Rate", f"{summary.get('acceptance_rate', 0.0):.2%}")
        c3.metric("Avg Decision Time", f"{summary.get('avg_decision_time_seconds', 0.0):.2f}s")


def main() -> None:
    st.set_page_config(page_title="Supply Chain FL + LLM Dashboard", layout="wide")
    st.title("Privacy-Preserving Human-Centric Supply Chain Analytics")

    with st.sidebar:
        st.header("Connection")
        api_base = st.text_input("API Base URL", value=DEFAULT_API_URL)
        health = _safe_get(api_base, "/health")
        if health:
            st.success("API reachable")
        else:
            st.warning("API not reachable")

    render_fl_metrics(api_base)
    forecasts_payload = render_forecasts(api_base)
    optimization = render_optimization(api_base, forecasts_payload)
    llm_payload = render_llm(api_base, optimization)
    render_feedback(api_base, llm_payload)


if __name__ == "__main__":
    main()
