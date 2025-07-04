from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    from langchain_community.llms import FakeListLLM
import os

load_dotenv()

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )
except Exception:
    llm = FakeListLLM(responses=["""# Research Report: Cross-Sell and Upsell Opportunities for C001
## Introduction
This report analyzes recent purchasing behavior and benchmarks against industry peers.

## Customer Overview
- Industry: Electronics
- Annual Revenue: $139000000
- Recent Purchases: Drill Bits, Protective Gloves, Generators, Workflow Automation

## Data Analysis
- Purchase patterns: Frequent Purchases: Drill Bits, Protective Gloves, Generators, Workflow Automation
- Industry peers commonly purchase: Collaboration Suite, API Integrations, Advanced Analytics

## Recommendations
1. Collaboration Suite (Cross-sell opportunity based on industry patterns)
2. API Integrations (Cross-sell opportunity based on industry patterns)
3. Advanced Analytics (Cross-sell opportunity based on industry patterns)
4. Backup Batteries (Upsell opportunity based on product affinity)
5. Power Cords (Upsell opportunity based on product affinity)
6. Drills (Upsell opportunity based on product affinity)
7. Safety Gear (Upsell opportunity based on product affinity)

## Conclusion
Targeted campaigns focusing on these products can increase revenue and customer satisfaction."""] * 10)

# Prompt for research report
report_prompt = PromptTemplate(
    input_variables=["customer_id", "industry", "revenue", "recent_purchases", "pattern_summary", "missing_products", "recommendations"],
    template="""
Generate a cross-sell and upsell report in markdown format:
# Research Report: Cross-Sell and Upsell Opportunities for {customer_id}
## Introduction
This report analyzes recent purchasing behavior and benchmarks against industry peers.

## Customer Overview
- Industry: {industry}
- Annual Revenue: ${revenue}
- Recent Purchases: {recent_purchases}

## Data Analysis
- Purchase patterns: {pattern_summary}
- Industry peers commonly purchase: {missing_products}

## Recommendations
{recommendations}

## Conclusion
Targeted campaigns focusing on these products can increase revenue and customer satisfaction.

Output only the report in the specified markdown format. Do not include critiques, suggestions for improvement, or any additional commentary.
"""
)

def generate_research_report(customer_profile: Dict[str, Any], purchase_patterns: Dict[str, Any], scored_opportunities: List[Dict[str, Any]]) -> str:
    """
    Generates a natural-language research report with recommendations.
    """
    try:
        if "error" in customer_profile or "error" in purchase_patterns:
            return "Error: Unable to generate report due to missing data"

        # Prepare data for the prompt
        pattern_summary = f"Frequent Purchases: {', '.join(purchase_patterns.get('frequent_products', []))}"
        missing_products = ", ".join(purchase_patterns.get("missing_products", []))
        recommendations = "\n".join(
            [f"{i+1}. {opp['product']} ({opp.get('rationale', 'No rationale provided')})" for i, opp in enumerate(scored_opportunities)]
        ) if scored_opportunities else "No recommendations available due to lack of scored opportunities."

        # Generate report using LangChain
        report = llm.invoke(
            report_prompt.format(
                customer_id=customer_profile.get("customer_id", "Unknown"),
                industry=customer_profile.get("industry", "Unknown"),
                revenue=customer_profile.get("revenue", 0.0),
                recent_purchases=", ".join(customer_profile.get("recent_purchases", [])),
                pattern_summary=pattern_summary,
                missing_products=missing_products,
                recommendations=recommendations
            )
        ).content.strip()
        return report

    except Exception as e:
        return f"Failed to generate report: {str(e)}"