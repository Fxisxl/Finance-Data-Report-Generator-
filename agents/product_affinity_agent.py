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
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )
except Exception:
    llm = FakeListLLM(responses=['["Backup Batteries", "Power Cords", "API Integrations", "Advanced Analytics"]'] * 10)

# Prompt for related products
affinity_prompt = PromptTemplate(
    input_variables=["products"],
    template="""
Given the following frequently purchased products: {products}, suggest related or co-purchased products based on general product purchasing behavior.
Return the result strictly in JSON array format, like:
["related_product1", "related_product2"]
Only return the JSON array.
"""
)

def suggest_related_products(frequent_products: List[str]) -> List[str]:
    """
    Suggests related/co-purchased products based on a list of frequent products.
    Combines static affinity map with Gemini-generated suggestions.
    """
    try:
        if not frequent_products:
            return []

        # Static rule-based affinities for robustness
        affinity_map = {
            "Generators": ["Backup Batteries", "Power Cords"],
            "Drills": ["Drill Bits", "Protective Gloves"],
            "Advanced Analytics": ["API Integrations", "Workflow Automation"],
            "Collaboration Suite": ["Advanced Analytics", "Workflow Automation"],
            "Workflow Automation": ["AI Insights Module", "API Integrations"],
            "Drill Bits": ["Drills", "Protective Gloves"],
            "Protective Gloves": ["Safety Gear", "Drills"]
        }

        static_suggestions = []
        for product in frequent_products:
            static_suggestions.extend(affinity_map.get(product, []))

        # Gemini-generated suggestions
        gemini_response = llm.invoke(
            affinity_prompt.format(products=str(frequent_products))
        ).content.strip()

        dynamic_suggestions = []
        try:
            dynamic_suggestions = json.loads(gemini_response)
            if not isinstance(dynamic_suggestions, list):
                dynamic_suggestions = []
        except json.JSONDecodeError:
            dynamic_suggestions = []

        # Combine static + Gemini, remove duplicates
        return list(set(static_suggestions + dynamic_suggestions))

    except Exception as e:
        return []