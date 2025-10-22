import requests
from typing import Optional


class RAGClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def ask(self, query: str) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        r = requests.post(
            f"{self.base_url}/ask",
            json={"query": query},
            headers=headers,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
