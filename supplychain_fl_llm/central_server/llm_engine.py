from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from utils.logging_utils import utc_now_iso


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "local-model"
    api_key: str = "lm-studio"
    temperature: float = 0.1


class LLMDecisionEngine:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._llm = ChatOpenAI(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
        )

    @staticmethod
    def _build_summary(optimization_output: dict[str, Any]) -> str:
        totals = optimization_output.get("totals", {})
        recommendations = optimization_output.get("recommendations", [])

        top = sorted(recommendations, key=lambda item: item.get("reorder_level", 0.0), reverse=True)[:5]
        lines = [
            f"Total cost: {totals.get('total_cost', 0.0):.2f}",
            f"Service level: {totals.get('service_level', 0.0):.4f}",
            f"Emissions: {totals.get('emissions', 0.0):.2f}",
            "Top product actions:",
        ]
        for item in top:
            lines.append(
                "- Product {pid}: reorder={reorder:.2f}, primary={primary:.2f}, "
                "backup={backup:.2f}, service={service:.3f}".format(
                    pid=item.get("product_id"),
                    reorder=float(item.get("reorder_level", 0.0)),
                    primary=float(item.get("source_primary", 0.0)),
                    backup=float(item.get("source_backup", 0.0)),
                    service=float(item.get("service_level", 0.0)),
                )
            )
        return "\n".join(lines)

    @staticmethod
    def _rank_recommendations(optimization_output: dict[str, Any]) -> list[dict[str, Any]]:
        ranked = sorted(
            optimization_output.get("recommendations", []),
            key=lambda item: (
                float(item.get("projected_shortage", 0.0)),
                float(item.get("reorder_level", 0.0)),
            ),
            reverse=True,
        )
        return ranked[:5]

    def _fallback_explanation(self, optimization_output: dict[str, Any], manager_question: str) -> str:
        totals = optimization_output.get("totals", {})
        ranked = self._rank_recommendations(optimization_output)
        if not ranked:
            return "No optimization actions are available yet. Run optimization before requesting explanation."

        top = ranked[0]
        return (
            "Fallback analytical explanation (LLM unavailable):\n"
            f"- Highest-priority product is {top.get('product_id')} because projected shortage pressure and reorder need are largest.\n"
            f"- Current plan targets service level {totals.get('service_level', 0.0):.3f} while balancing total cost "
            f"{totals.get('total_cost', 0.0):.2f} and emissions {totals.get('emissions', 0.0):.2f}.\n"
            "- Recommendation: increase inventory buffer for high-risk products, allocate more to primary sourcing where possible, "
            "and reserve backup sourcing for disruption-sensitive demand."
            f"\nQuestion addressed: {manager_question}"
        )

    def generate_recommendations(
        self,
        optimization_output: dict[str, Any],
        manager_question: str,
    ) -> dict[str, Any]:
        summary = self._build_summary(optimization_output)
        ranked = self._rank_recommendations(optimization_output)

        template = ChatPromptTemplate.from_template(
            """
You are a supply chain decision intelligence assistant.
You only receive aggregated optimization outputs and must not infer raw partner data.

Manager question:
{question}

Optimization summary:
{summary}

Tasks:
1) Explain the main tradeoffs across cost, service level, and emissions.
2) Provide a ranked recommendation list with managerial language.
3) Keep response concise and actionable.
"""
        )

        try:
            chain = template | self._llm
            response = chain.invoke({"question": manager_question, "summary": summary})
            explanation = response.content if hasattr(response, "content") else str(response)
            mode = "llm"
        except Exception:
            explanation = self._fallback_explanation(optimization_output, manager_question)
            mode = "fallback"

        return {
            "timestamp": utc_now_iso(),
            "mode": mode,
            "manager_question": manager_question,
            "summary": summary,
            "explanation": explanation,
            "ranked_recommendations": ranked,
        }


def build_llm_engine_from_env() -> LLMDecisionEngine:
    config = LLMConfig(
        base_url=os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        model=os.getenv("LM_STUDIO_MODEL", "local-model"),
        api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        temperature=float(os.getenv("LM_STUDIO_TEMPERATURE", "0.1")),
    )
    return LLMDecisionEngine(config)
