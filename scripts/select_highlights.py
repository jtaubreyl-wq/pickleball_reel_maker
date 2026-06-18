import json
from pathlib import Path
from typing import List

from src.highlights.models import Rally
from src.highlights.highlight_selector import (
    select_top_highlights,
    export_rallies_to_dict,
)


def load_rallies_from_json(path: Path) -> List[Rally]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rallies: List[Rally] = []
    for item in data:
        rallies.append(
            Rally(
                rally_id=item["rally_id"],
                start_time=item["start_time"],
                end_time=item["end_time"],
                highlight_score=item["highlight_score"],
                metadata=item.get("metadata"),
            )
        )
    return rallies


def main(
    input_path: str = "artifacts/rallies.json",
    output_path: str = "artifacts/selected_highlights.json",
    max_highlights: int = 10,
):
    input_file = Path(input_path)
    output_file = Path(output_path)

    rallies = load_rallies_from_json(input_file)
    selected = select_top_highlights(rallies, max_highlights=max_highlights)

    export_data = export_rallies_to_dict(selected)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    print(f"Exported {len(selected)} highlights to {output_file}")


if __name__ == "__main__":
    main()
