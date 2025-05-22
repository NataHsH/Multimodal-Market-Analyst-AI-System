# data_extraction_chromaDB.py

import os
import sys
import pandas as pd
import chromadb
from datetime import datetime
from langchain.tools import tool

# Define and register chroma_text path
chroma_text_path = os.path.join(os.getcwd(), 'agents', 'chroma_text')
sys.path.insert(0, chroma_text_path)

@tool("load_company_data_from_chromadb", return_direct=False)
def load_company_data(company_name: str, years: int = 5) -> pd.DataFrame:
    """
    Load historical stock price data from ChromaDB for the specified company and number of past years.
    Returns a DataFrame with columns: ['ds', 'y', 'company'].
    """
    current_year = datetime.now().year
    start_year = current_year - years
    all_data = []

    client = chromadb.Client()
    collection = client.get_collection('chroma_text')

    for year in range(start_year, current_year):
        results = collection.query(
            query_texts=[f"{company_name} stock_price"],
            n_results=1000,
            where={"company": company_name, "data_type": "stock_price", "year": str(year)}
        )

        dates = []
        prices = []

        for metadata, document in zip(results['metadatas'], results['documents']):
            date_val = metadata.get('date')
            try:
                price_val = float(document)
            except (ValueError, TypeError):
                price_val = None

            if date_val and price_val is not None:
                dates.append(date_val)
                prices.append(price_val)

        df = pd.DataFrame({
            'ds': pd.to_datetime(dates),
            'y': prices,
            'company': company_name
        })

        all_data.append(df)

    if not all_data:
        raise ValueError(f"No data found for {company_name}.")

    return pd.concat(all_data).reset_index(drop=True)
