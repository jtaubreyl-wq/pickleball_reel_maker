from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from src.highlights.models import Rally
from src.highlights.highlight_selector import (
    select_top_highlights,
    export_rallies_to_dict,
)

app = FastAPI(title="Pickleball Highlight API")


class RallyIn(BaseModel):
    rally_id: str
    start_time: float
    end_time: float
    highlight_score: float
    metadata: Optional[dict] = None


class SelectRequest(BaseModel):
    rallies: List[RallyIn]
    max_highlights: int = 10


@app.post("/select-highlights")
def select_highlights_api(req: SelectRequest):
    rallies = [
        Rally(
            rally_id=r.rally_id,
            start_time=r.start_time,
            end_time=r.end_time,
            highlight_score=r.highlight_score,
            metadata=r.metadata,
        )
        for r in req.rallies
    ]

    selected = select_top_highlights(rallies, max_highlights=req.max_highlights)
    return {"selected": export_rallies_to_dict(selected)}