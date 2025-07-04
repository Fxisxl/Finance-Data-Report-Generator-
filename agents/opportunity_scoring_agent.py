from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    from langchain_community.llms import FakeListLLM
import os
import json

load_dotenv()

# Initialize Gemini model
# try:
llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )
# except Exception:
#     llm = FakeListLLM(responses=['[{"product": "Collaboration Suite", "score": 0.8, "rationale": "Cross-sell opportunity based on industry patterns"}, {"product": "API Integrations", "score": 0.8, "rationale": "Cross-sell opportunity based on industry patterns"}, {"product": "Advanced Analytics", "score": 0.8, "rationale": "Cross-sell opportunity based on industry patterns"}, {"product": "Backup Batteries", "score": 0.6, "rationale": "Upsell opportunity based on product affinity"}, {"product": "Power Cords", "score": 0.6, "rationale": "Upsell opportunity based on product affinity"}, {"product": "Drills", "score": 0.6, "rationale": "Upsell opportunity based on product affinity"}, {"product": "Safety Gear", "score": 0.6, "rationale": "Upsell opportunity based on product affinity"}]'] * 10)

# Prompt for opportunity scoring
scoring_prompt = PromptTemplate(
    input_variables=["missing_products", "affinity_products", "purchased_products"],
    template="""
Score cross-sell and upsell opportunities for the following products.
Missing products: {missing_products}
Affinity products: {affinity_products}
Purchased products: {purchased_products}

Return output strictly in JSON array format like:
[
  {"product": "product_name", "score": float (0.0 - 1.0), "rationale": "reason for score"},
  ...
]
Only return the JSON array. Do not include any explanations or additional text.
"""
)

def score_opportunities(missing_products: List[str], affinity_products: List[str], purchased_products: List[str] = None) -> List[Dict[str, Any]]:
    """
    Scores cross-sell and upsell opportunities with rationale.
    """
    try:
        if purchased_products is None:
            purchased_products = []
        opportunities = list(set(missing_products + affinity_products) - set(purchased_products))  # Exclude purchased products
        if not opportunities:
            return []

        # Fallback static scoring
        fallback = []
        for product in opportunities:
            score = 0.8 if product in missing_products else 0.6
            rationale = f"{'Cross-sell' if product in missing_products else 'Upsell'} opportunity based on {'industry patterns' if product in missing_products else 'product affinity'}"
            fallback.append({"product": product, "score": score, "rationale": rationale})

        # Gemini-generated scoring
        result = llm.invoke(
            scoring_prompt.format(
                missing_products=str(missing_products),
                affinity_products=str(affinity_products),
                purchased_products=str(purchased_products)
            )
        ).content.strip()

        try:
            parsed = json.loads(result)
            if isinstance(parsed, list) and all(isinstance(item, dict) and "product" in item and "score" in item and "rationale" in item for item in parsed):
                # Filter out purchased products from LLM response
                return [item for item in parsed if item["product"] not in purchased_products]
            return fallback
        except json.JSONDecodeError:
            return fallback

    except Exception:
        return fallback