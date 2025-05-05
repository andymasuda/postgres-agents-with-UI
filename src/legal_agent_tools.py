import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import datetime
import json
from typing import Any, Callable, Set
from sqlalchemy import create_engine
from azure.ai.projects.telemetry import trace_function
from opentelemetry import trace
from pathlib import Path
import logging
import sys

# Redirect debug prints to Azure log stream (stdout)
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

CONN_STR = os.getenv("SQLALCHEMY_PG_CONNECTION")
logger.debug(f"Loaded CONN_STR: {CONN_STR}")

# The trace_func decorator will trace the function call and enable adding additional attributes
# to the span in the function implementation. Note that this will trace the function parameters and their values.

# Get data from the Postgres database
@trace_function()
def hybrid_search_cases(query: str, limit: int = 10) -> str:
    """
    Fetches document chunks relevant to the specified query using hybrid search
    (combining full-text search and vector search with RRF scoring).

    :param query: The query to search for in the document embeddings and full-text search.
    :type query: str
    :param limit: The maximum number of document chunks to fetch, defaults to 10.
    :type limit: int, optional

    :return: Document chunks as a JSON string.
    :rtype: str
    """
    db = create_engine(CONN_STR)
    logger.debug("Database engine created successfully. Connection is ready to use.")

    hybrid_query = """
    WITH full_text_results AS (
        SELECT id, chunk, 
                ts_rank(tsvector_column, plainto_tsquery('english', %s)) AS rank
        FROM document_chunks
        WHERE tsvector_column @@ plainto_tsquery('english', %s)
        LIMIT %s
    ),
    vector_results AS (
        SELECT id, chunk, 
                embeddings::vector <=> azure_openai.create_embeddings(
                'text-embedding-3-small', %s)::vector AS similarity
        FROM document_chunks
        ORDER BY similarity ASC
        LIMIT %s
    ),
    combined_results AS (
        SELECT 
            COALESCE(ft.id, vt.id) AS id,
            COALESCE(ft.chunk, vt.chunk) AS chunk,
            ft.rank,
            vt.similarity
        FROM full_text_results ft
        FULL OUTER JOIN vector_results vt
        ON ft.id = vt.id
    )
    SELECT id, chunk, 
            1 / (0.5 + ROW_NUMBER() OVER (ORDER BY rank DESC NULLS LAST)) AS full_text_score,
            1 / (0.5 + ROW_NUMBER() OVER (ORDER BY similarity ASC NULLS LAST)) AS vector_score,
            (1 / (0.5 + ROW_NUMBER() OVER (ORDER BY rank DESC NULLS LAST)) +
            1 / (0.5 + ROW_NUMBER() OVER (ORDER BY similarity ASC NULLS LAST))) AS rrf_score
    FROM combined_results
    ORDER BY rrf_score DESC
    LIMIT %s;
    """

    # Execute the hybrid query
    df = pd.read_sql(hybrid_query, db, params=(query, query, limit, query, limit, limit))

    # Debugging: Log the query and returned data
    logger.debug("Executed hybrid SQL query:")
    logger.debug(hybrid_query)
    logger.debug("Returned data:")
    logger.debug(df)

    span = trace.get_current_span()
    span.set_attribute("requested_query", hybrid_query)

    documents_json = json.dumps(df.to_json(orient="records"))
    span.set_attribute("documents_json", documents_json)

    # Log the JSON before returning
    logger.debug("Generated JSON:")
    logger.debug(documents_json)

    logger.debug("Executing hybrid_search_cases successfully.")
    return documents_json
'''
@trace_function()
def vector_search_cases(vector_search_query: str, limit: int = 10) -> str:
    """
    Fetches document chunks relevant to the specified query.

    :param query: The query to search for in the document embeddings.
    :type query: str
    :param limit: The maximum number of document chunks to fetch, defaults to 10.
    :type limit: int, optional

    :return: Document chunks as a JSON string.
    :rtype: str
    """
    db = create_engine(CONN_STR)

    query = """
    SELECT id, chunk, 
    embeddings::vector <=> azure_openai.create_embeddings(
    'text-embedding-3-small', %s)::vector as similarity
    FROM document_chunks
    ORDER BY similarity
    LIMIT %s;
    """
    
    # Fetch cases information from the database
    df = pd.read_sql(query, db, params=(vector_search_query, limit))

    # Debugging: Log the query and returned data
    logger.debug("Executed SQL query:")
    logger.debug(query)
    logger.debug("Returned data:")
    logger.debug(df)

    span = trace.get_current_span()
    span.set_attribute("requested_query", query)

    documents_json = json.dumps(df.to_json(orient="records"))
    span.set_attribute("documents_json", documents_json)

    # Log the JSON before returning
    logger.debug("Generated JSON:")
    logger.debug(documents_json)
  
@trace_function()
def count_cases(vector_search_query: str, limit: int = 10) -> str:
    """
    Count the number of document chunks related to the specified query.

    :param query: The query to search for in the document embeddings.
    :type query: str
    :param limit: The maximum number of document chunks to count, defaults to 10.
    :type limit: int, optional

    :return: Count information as a JSON string.
    :rtype: str
    """
    db = create_engine(CONN_STR)

    query = """
    SELECT COUNT(*) 
    FROM document_chunks
    WHERE embeddings::vector <=> azure_openai.create_embeddings(
        'text-embedding-3-small', 
    %s)::vector < 0.8 -- 0.8 is the threshold for similarity
    LIMIT %s;
    """
    
    # Debugging: Log the query and parameters
    logger.debug("Executing SQL query:")
    logger.debug(query)
    logger.debug("With parameters:")
    logger.debug((vector_search_query, limit))  # Log the parameters being passed

    df = pd.read_sql(query, db, params=(vector_search_query, limit))  # Ensure params match placeholders

    logger.debug("Returned data:")
    logger.debug(df)

    span = trace.get_current_span()
    span.set_attribute("requested_query", query)
    documents_count = json.dumps(df.to_json(orient="records"))
    span.set_attribute("result", documents_count)

    # Log the JSON before returning
    logger.debug("Generated JSON:")
    logger.debug(documents_count)

    return documents_count
'''
# Statically defined user functions for fast reference
user_functions: Set[Callable[..., Any]] = {
    hybrid_search_cases
}