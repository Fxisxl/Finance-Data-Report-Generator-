import pandas as pd
import os
from typing import Dict, Any
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    from langchain_community.llms import FakeListLLM

load_dotenv()

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )
except Exception:
    llm = FakeListLLM(responses=["{'customer_id': 'C001', 'customer_name': 'Edge Communications', 'industry': 'Electronics', 'revenue': 139000000.0, 'number_of_employees': 1000, 'location': 'Austin, TX, USA', 'recent_purchases': ['Drill Bits', 'Protective Gloves', 'Generators', 'Workflow Automation']}"] * 10)

# Prompt to format customer profile
profile_prompt = PromptTemplate(
    input_variables=["customer_data"],
    template="Format the following customer data into a structured profile:\n{customer_data}\nOutput a JSON-like dictionary with customer_id, customer_name, industry, revenue, number_of_employees, location, recent_purchases."
)

def get_customer_profile(customer_id: str) -> Dict[str, Any]:
    """
    Extracts a customer profile dict from CSV by customer_id.
    Uses LangChain to format the output.
    """
    try:
        data_file = os.getenv("DATA_FILE", "data/customer_data_purchases.csv")
        if not os.path.exists(data_file):
            return {"error": f"CSV file {data_file} not found"}

        df = pd.read_csv(data_file)
        customer_rows = df[df["Customer ID"] == customer_id]

        if customer_rows.empty:
            return {"error": f"Customer ID {customer_id} not found"}

        first = customer_rows.iloc[0]
        profile = {
            "customer_id": customer_id,
            "customer_name": first.get("Customer Name", "Unknown"),
            "industry": first.get("Industry", "Unknown"),
            "revenue": first.get("Annual Revenue (USD)", 0.0),
            "number_of_employees": first.get("Number of Employees", 0),
            "location": first.get("Location", "Unknown"),
            "recent_purchases": list(customer_rows["Product"].unique())
        }

        # Use LangChain to format
        formatted_profile = llm.invoke(
            profile_prompt.format(customer_data=str(profile))
        ).content
        try:
            return eval(formatted_profile)
        except:
            return profile

    except Exception as e:
        return {"error": f"Failed to retrieve profile: {str(e)}"}