import pandas as pd
import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    from langchain_community.llms import FakeListLLM
from langchain.prompts import PromptTemplate

load_dotenv()

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )
except Exception:
    llm = FakeListLLM(responses=['{"frequent_products": ["Drill Bits", "Protective Gloves", "Generators", "Workflow Automation"], "missing_products": ["Collaboration Suite", "API Integrations"]}'] * 10)

DATA_FILE = os.getenv("DATA_FILE", "data/customer_data_purchases.csv")

pattern_prompt = PromptTemplate(
    input_variables=["products", "industry"],
    template="""
Given the following products: {products} purchased by a customer in the {industry} industry, suggest frequent and missing products based on common industry patterns.
Return output strictly in JSON format with:
{{
  "frequent_products": ["product1", "product2"],
  "missing_products": ["product3", "product4"]
}}
Only return the JSON.
"""
)

def analyze_purchase_patterns(customer_id: str) -> Dict[str, Any]:
    """
    Analyzes purchase patterns to identify frequent and missing products.
    """
    try:
        if not os.path.exists(DATA_FILE):
            return {"error": f"CSV file {DATA_FILE} not found"}
        df = pd.read_csv(DATA_FILE)
        cust = df[df["Customer ID"] == customer_id]
        if cust.empty:
            return {"error": f"Customer ID {customer_id} not found"}

        products = cust["Product"].tolist()
        industry = cust.iloc[0]["Industry"]

        # Fallback static analysis
        industry_patterns = {
            "Electronics": ["Collaboration Suite", "API Integrations", "Advanced Analytics"],
            "Apparel": ["Advanced Analytics", "Workflow Automation"],
            "Hospitality": ["API Integrations", "Workflow Automation"],
            "Construction": ["Backup Batteries", "Safety Gear", "Heavy Equipment"],
            "Energy": ["AI Insights Module", "Collaboration Suite"]
        }
        frequent_products = list(set(products))
        missing_products = [p for p in industry_patterns.get(industry, []) if p not in products]

        # Try Gemini analysis
        analysis = llm.invoke(
            pattern_prompt.format(products=str(products), industry=industry)
        ).content.strip()

        try:
            result = json.loads(analysis)
            if isinstance(result, dict) and "frequent_products" in result and "missing_products" in result:
                return result
        except json.JSONDecodeError:
            pass

        # Return fallback
        return {
            "frequent_products": frequent_products,
            "missing_products": missing_products
        }

    except Exception as e:
        return {"error": f"Failed to analyze patterns: {str(e)}"}