# data_extraction_chromaDB.py

import pandas as pd
import chromadb

import sys
sys.path.insert(0, 'C:\\path\\to\\project_root\\chroma_text')

import os
print(os.path.exists('C:\\path\\to\\project_root\\chroma_text'))


def extract_data_from_chromadb(company_name: str, data_type: str, time_frame: str) -> pd.DataFrame:
    """
    Extract data from ChromaDB collection for a given company, data type, and year (time_frame).
    Returns a DataFrame with columns: ['date', 'stock_price', 'company'].
    """
    client = chromadb.Client()
    collection = client.get_collection('financial_data')  # Убедись, что имя коллекции правильное

    results = collection.query(
        query_texts=[f"{company_name} {data_type}"],
        n_results=1000,
        where={"company": company_name, "data_type": data_type, "year": time_frame}
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
        'date': pd.to_datetime(dates),
        'stock_price': prices,
        'company': company_name
    })

    return df
