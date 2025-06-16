import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import datetime
import json
from typing import Any, Callable, Set
from sqlalchemy import create_engine
from azure.ai.agents.telemetry import trace_function
from opentelemetry import trace
from openai import OpenAI
from pathlib import Path
import logging
import sys
import requests
import time

# Redirect debug prints to Azure log stream (stdout)
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

class NoInfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.INFO

logger.addFilter(NoInfoFilter())

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

CONN_STR = os.getenv("SQLALCHEMY_PG_CONNECTION")
logger.debug(f"Loaded CONN_STR: {CONN_STR}")

# The trace_func decorator will trace the function call and enable adding additional attributes
# to the span in the function implementation. Note that this will trace the function parameters and their values.

@trace_function()
def sql_search(user_query: str) -> str:
    start = time.time()
    logger.debug("sql_search started")
    """
    Converts a user query into an SQL query, executes it, and returns the results as JSON.

    :param user_query: The user's natural language query.
    :type user_query: str

    :return: Query results as a JSON string.
    :rtype: str
    """
    API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    DEPLOYMENT = os.getenv("MODEL_DEPLOYMENT_NAME")

    db = create_engine(CONN_STR)
    logger.debug("Database engine created successfully. Connection is ready to use.")

    # Updated system prompt for full-text search using tsvector
    system_prompt = (
        "You are an expert assistant that converts natural language questions into efficient, read-only SQL queries for a PostgreSQL database. The table is named 'invoices'.\n\n"
        "**IMPORTANT SEARCH RULE:**\n"
        "To search for text, keywords, or descriptions (like product names, customer names, etc.), you MUST use the special full-text search column named `tsv`. This column contains all searchable text from the invoice. The correct way to use it is with the `@@ to_tsquery()` operator. For example, to find 'cedar', the query is `WHERE tsv @@ to_tsquery('english', 'cedar')`. To find 'cedar' and 'panels', it is `WHERE tsv @@ to_tsquery('english', 'cedar & panels')`.\n\n"
        "**DO NOT GUESS a text column and use `=` or `ILIKE`. This is wrong and will fail.** Always use `tsv @@ to_tsquery()` for any text search.\n\n"
        "Use standard `WHERE` clauses only for filtering on columns with exact, structured data like \"Region\", \"Channel\", \"Product Type\", \"Major Code\", or for numeric/date ranges on \"Sales\" or \"Invoice Date\".\n\n"
        "TABLE SCHEMA:\n"
        "ID: Unique identifier for each invoice (the invoice number).\n"
        "\"FiscalWeekBeginDate\", \"Invoice Date\": Date columns.\n"
        "\"Region\": The sales region (e.g., 'Central'). A structured category.\n"
        "\"Facility Name\": The name of the facility or branch (e.g., 'Birmingham'). A structured category.\n"
        "\"Branch Id\": The short code for the facility (e.g., 'BIR'). A structured category.\n"
        "\"Channel\": The sales channel (e.g., 'Direct', 'Warehouse'). A structured category.\n"
        "\"soldto_name\", \"shipto_name\": Customer name information. Search these using the `tsv` column.\n"
        "\"Product Type\": The high-level product category (e.g., 'Specialty'). A structured category.\n"
        "\"Major Code\": Numeric code for the major product category, stored as text. When filtering, always compare as a string and put quotes around the number (e.g., WHERE \"Major Code\" = '1').\n"
        "\"Mid Code\": Numeric code for the mid-level product category, stored as text. When filtering, always compare as a string and put quotes around the number (e.g., WHERE \"Mid Code\" = '2').\n"
        "\"Minor Code\": Three-letter code for the minor product category (e.g., 'OSB', 'CED').\n"
        "\"Major Desc\", \"Mid Desc\", \"Minor Desc\", \"Item Desc\": Text descriptions for product categories and items. Search these using the `tsv` column.\n"
        "\"Item\": The unique identifier for the item/product that was sold. This is the item ID, not the invoice ID.\n"
        "\"Sales\", \"Gross Profit\": Numeric columns for financial data.\n"
        "\"GM Percent\", \"TLE\": Numeric columns representing percentages or ratios, stored as decimals.\n"
        "tsv: A special tsvector column for fast text searching. USE THIS FOR ALL TEXT AND KEYWORD SEARCHES.\n\n"
        "RULES:\n"
        "1. For text searches (customers, products, descriptions), ALWAYS use `WHERE tsv @@ to_tsquery(...)`.\n"
        "2. For filtering on structured categories (\"Region\", \"Channel\", \"Facility Name\", \"Major Code\") or numbers (\"Sales\", \"Gross Profit\"), use standard operators (`=`, `>`, `<`). For \"Major Code\" and \"Mid Code\", always compare as a string and put quotes around the number.\n"
        "3. ALWAYS use double quotes around column names (e.g., \"Sales\", \"Facility Name\").\n"
        "4. Do not use `SELECT *`; always specify the columns needed.\n"
        "5. Only generate SQL, no explanations.\n"
        "6. Remember: \"ID\" is the invoice number, and \"Item\" is the item/product ID."
    )

    url = f"{ENDPOINT}/openai/deployments/{DEPLOYMENT}/chat/completions?api-version=2024-02-15-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }
    data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "max_tokens": 256,
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    sql_query = response.json()["choices"][0]["message"]["content"].strip()
    # Remove code block markers if present
    if sql_query.startswith("```"):
        sql_query = sql_query.lstrip("`").lstrip("sql").lstrip().rstrip("`")
        sql_query = sql_query.replace("```", "").strip()
    logger.debug(f"Generated SQL query: {sql_query}")

    # Execute the generated SQL query
    try:
        df = pd.read_sql(sql_query, db)
        logger.debug("SQL query executed successfully.")
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        return json.dumps({"error": str(e)})

    # Convert results to JSON
    results_json = df.to_json(orient="records")
    logger.debug("Number of vector search results: %d", len(df))
    logger.debug("Query results as JSON:")
    logger.debug(results_json)
    logger.debug("Number of vector search results: %d", len(df))
    logger.debug(f"Generated SQL query: {sql_query}")

    span = trace.get_current_span()
    span.set_attribute("user_query", user_query)
    span.set_attribute("sql_query", sql_query)
    span.set_attribute("results_json", results_json)

    logger.debug(f"sql_search finished in {time.time() - start:.2f} seconds")
    return results_json

@trace_function()
def vector_search(vector_search_query: str, similarity_threshold: float = 0.6, limit: int = 10) -> str:
    start = time.time()
    logger.debug("vector_search started")
    """
    Fetches the top N most semantically similar invoice rows to the query,
    and includes the total count of all relevant rows (not just the top N).

    :param vector_search_query: The query to search for in the invoice embeddings.
    :type vector_search_query: str
    :param similarity_threshold: The similarity threshold for relevance (default 0.6).
    :type similarity_threshold: float, optional
    :param limit: The maximum number of rows to return in the output (default 10).
    :type limit: int, optional

    :return: JSON with 'results' (top N rows) and 'total_relevant_count'.
    :rtype: str
    """
    db = create_engine(CONN_STR)
    logger.debug("Database engine created successfully. Connection is ready to use.")

    # Compute the embedding ONCE and use it in the query
    embedding_query = """
    SELECT azure_openai.create_embeddings('text-embedding-3-small', %s)::vector AS query_embedding
    """
    try:
        embedding_df = pd.read_sql(embedding_query, db, params=(vector_search_query,))
        query_embedding = embedding_df.iloc[0]['query_embedding']
    except Exception as e:
        logger.error(f"Embedding computation error: {e}")
        return json.dumps({"error": str(e)})

    query = """
    WITH query_vec AS (
        SELECT %s::vector AS embedding
    ),
    filtered AS (
        SELECT
            "ID", "FiscalWeekBeginDate", "Invoice Date", "Region", "Facility Name", "Branch Id", "Channel",
            "soldto_name", "shipto_name", "Product Type", "Major Code", "Major Desc", "Mid Code", "Mid Desc",
            "Minor Code", "Minor Desc", "Item", "Item Desc", "Sales", "Gross Profit", "GM Percent", "TLE",
            embeddings::vector <=> query_vec.embedding AS similarity
        FROM invoices, query_vec
        WHERE (embeddings::vector <=> query_vec.embedding) < %s
        ORDER BY similarity
    )
    SELECT *, (SELECT COUNT(*) FROM filtered) AS total_relevant_count
    FROM filtered
    ;
    """

    logger.debug("Vector search SQL query:\n%s", query)

    try:
        df = pd.read_sql(query, db, params=(query_embedding, similarity_threshold, limit))
        logger.debug("Vector search SQL query executed successfully.")
    except Exception as e:
        logger.error(f"Vector search SQL execution error: {e}")
        return json.dumps({"error": str(e)})

    # Extract total count (will be the same for all rows, or 0 if no rows)
    total_count = int(df['total_relevant_count'].iloc[0]) if not df.empty else 0
    # Only return the top `limit` results (already limited in SQL)
    results = df.drop(columns=['total_relevant_count']).to_dict(orient="records")

    output = {
        "results": results,
        "total_relevant_count": total_count
    }

    logger.debug("Number of vector search results: %d", len(results))
    logger.debug("Total relevant count: %d", total_count)
    logger.debug("Vector search results as JSON:")
    logger.debug(json.dumps(output))

    span = trace.get_current_span()
    span.set_attribute("vector_search_query", vector_search_query)
    span.set_attribute("similarity_threshold", similarity_threshold)
    span.set_attribute("total_relevant_count", total_count)
    span.set_attribute("results_json", json.dumps(output))

    logger.debug(f"vector_search finished in {time.time() - start:.2f} seconds")
    return json.dumps(output)

# Statically defined user functions for fast reference
user_functions: Set[Callable[..., Any]] = {
    sql_search,
    vector_search,
}