"""Client for pulling MoRTH road accident stats from data.gov.in.

NOTE: fill in `accident_resource_id` in config/config.yaml once you've
located the exact dataset (search data.gov.in for "road accidents state
wise" -- the resource_id lives in the dataset's API page).
"""
import os
from typing import Any

from src.ingestion.base_client import BaseAPIClient


class DataGovInClient(BaseAPIClient):
    def __init__(self, resource_id: str):
        super().__init__(base_url="https://api.data.gov.in/resource")
        self.resource_id = resource_id
        self.api_key = os.environ.get("DATA_GOV_IN_API_KEY")
        if not self.api_key:
            raise RuntimeError("DATA_GOV_IN_API_KEY not set in environment (.env)")

    def fetch_records(self, offset: int = 0, limit: int = 100, filters: dict[str, str] | None = None) -> list[dict[str, Any]]:
        params = {
            "api-key": self.api_key,
            "format": "json",
            "offset": offset,
            "limit": limit,
        }
        if filters:
            for field, value in filters.items():
                params[f"filters[{field}]"] = value

        data = self.get(self.resource_id, params=params)
        return data.get("records", [])

    def fetch_all(self, filters: dict[str, str] | None = None, page_size: int = 100) -> list[dict[str, Any]]:
        """Paginate through all records for a given filter set."""
        records: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self.fetch_records(offset=offset, limit=page_size, filters=filters)
            if not page:
                break
            records.extend(page)
            offset += page_size
        return records
