"""Single place that knows how to connect to the warehouse.
Everything else (ingestion, validation, features) imports from here
rather than building its own connection string.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set. Copy .env.example to .env and fill it in.")
    return create_engine(url, pool_pre_ping=True)


def run_sql_file(path: str) -> None:
    """Execute a .sql file against the warehouse. Used for schema setup
    outside of the docker-compose auto-init path (e.g. in CI or after
    a manual schema change)."""
    with open(path, "r") as f:
        sql = f.read()
    engine = get_engine()
    with engine.begin() as conn:
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement:
                conn.exec_driver_sql(statement)
