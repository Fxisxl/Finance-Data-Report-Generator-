from fastapi import FastAPI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import Dict, Any, List
import os
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import agent functions
from agents.customer_context_agent import get_customer_profile
from agents.purchase_pattern_agent import analyze_purchase_patterns
from agents.product_affinity_agent import suggest_related_products
from agents.opportunity_scoring_agent import score_opportunities
from agents.recommendation_report_agent import generate_research_report

# Initialize FastAPI and environment
load_dotenv()
app = FastAPI(title="Cross-Sell API")

# Define state to pass between agent types
class AgentState(Dict):
    customer_id: str
    customer_profile: Dict[str, Any]
    purchase_patterns: Dict[str, Any]
    product_affinity: List[str]
    scored_opportunities: List[Dict[str, Any]]
    research_report: str

# Define LangGraph nodes
def customer_context_node(state: AgentState) -> Dict[str, Any]:
    logger.debug("Customer context node: Fetching profile for ID %s", state["customer_id"])
    profile = get_customer_profile(state["customer_id"])
    state["customer_profile"] = profile
    return state

def purchase_pattern_node(state: AgentState) -> Dict[str, Any]:
    logger.debug("Purchase pattern node: Analyzing patterns for customer ID %s", state["customer_profile"]["customer_id"])
    customer_id = state["customer_profile"]["customer_id"]
    patterns = analyze_purchase_patterns(customer_id)
    if "error" in patterns:
        logger.error("Purchase pattern error: %s", patterns["error"])
        raise ValueError(patterns["error"])
    state["purchase_patterns"] = patterns
    return state

def product_affinity_node(state: AgentState) -> Dict[str, Any]:
    logger.debug("Product affinity node: Generating suggestions for products %s", state["purchase_patterns"]["frequent_products"])
    related = suggest_related_products(state["purchase_patterns"]["frequent_products"])
    state["product_affinity"] = related
    return state

def opportunity_scoring_node(state: AgentState) -> Dict[str, Any]:
    logger.debug("Opportunity scoring node: Scoring %s and %s", state["purchase_patterns"]["missing_products"], state["product_affinity"])
    scored = score_opportunities(
        state["purchase_patterns"]["missing_products"],
        state["product_affinity"],
        state["customer_profile"].get("recent_purchases", [])  # Pass purchased products
    )
    state["scored_opportunities"] = scored
    return state

def recommendation_report_node(state: AgentState) -> Dict[str, Any]:
    logger.debug("Recommendation report node: Generating report for customer ID %s", state["customer_id"])
    report = generate_research_report(
        state["customer_profile"],
        state["purchase_patterns"],
        state["scored_opportunities"]
    )
    state["research_report"] = report
    return state

# Define the workflow
workflow = StateGraph(AgentState)
workflow.add_node("customer_context", customer_context_node)
workflow.add_node("purchase_pattern", purchase_pattern_node)
workflow.add_node("product_affinity_node", product_affinity_node)
workflow.add_node("opportunity_scoring", opportunity_scoring_node)
workflow.add_node("recommendation_report", recommendation_report_node)

# Define edges
workflow.add_edge("customer_context", "purchase_pattern")
workflow.add_edge("purchase_pattern", "product_affinity_node")
workflow.add_edge("product_affinity_node", "opportunity_scoring")
workflow.add_edge("opportunity_scoring", "recommendation_report")
workflow.add_edge("recommendation_report", END)

# Set entry point and compile
workflow.set_entry_point("customer_context")
graph = workflow.compile()

# FastAPI endpoint
@app.get("/recommendation")
async def recommendation(customer_id: str) -> Dict[str, Any]:
    try:
        print(customer_id)
        logger.info("Received request for customer_id: %s", customer_id)
        # Initialize state and run
        state = AgentState(customer_id=customer_id)
        result = await graph.ainvoke(state)


        logger.debug("Pipeline result: %s", result)
        if "error" in result["customer_profile"]:
            logger.error("Customer profile error: %s", result["customer_profile"]["error"])
            return {"error": result["customer_profile"]["error"]}
        if "error" in result["purchase_patterns"]:
            logger.error("Purchase pattern error: %s", result["purchase_patterns"]["error"])
            return {"error": result["purchase_patterns"]["error"]}

        # Return response
        logger.info("Generated report successfully for customer_id: %s", customer_id)
        return {
            "research_report": result["research_report"],
            "recommendations": result["scored_opportunities"]
        }
    except Exception as e:
        logger.error("Pipeline failed: %s", str(e))
        return {"error": f"Pipeline failed: {str(e)}"}