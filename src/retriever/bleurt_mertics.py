import httpx
from typing import List

class Bleurt20Client:
    def __init__(self, base_url: str = "http://localhost:8088", timeout: float = 60.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def score_batch(self, candidates: List[str], references: List[str]) -> List[float]:
        url = f"{self.base}/score"
        payload = {"candidates": candidates, "references": references}
        with httpx.Client(timeout=self.timeout) as cx:
            r = cx.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return [float(x) for x in data["scores"]]

    def score(self, candidate: str, reference: str) -> float:
        return self.score_batch([candidate], [reference])[0]