from sqlalchemy import create_engine, text
import pandas as pd

def read_from_db(
    database_url: str,
    query: str,
    params: dict | None = None
) -> pd.DataFrame:
    """
    Reads data from a database and returns a Pandas DataFrame.

    Args:
        database_url (str): SQLAlchemy DB URL
            Example:
            postgresql://user:password@localhost:5432/dbname
        query (str): SQL query
        params (dict): Optional query parameters

    Returns:
        pd.DataFrame
    """
    engine = create_engine(database_url)

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    return df
