import json
from pathlib import Path
import click

from src.highlights.models import Rally
from src.highlights.highlight_selector import (
    select_top_highlights,
    export_rallies_to_dict,
)


def load_rallies(path: Path):
    data = json.loads(path.read_text())
    return [
        Rally(
            rally_id=item["rally_id"],
            start_time=item["start_time"],
            end_time=item["end_time"],
            highlight_score=item["highlight_score"],
            metadata=item.get("metadata"),
        )
        for item in data
    ]


@click.group()
def cli():
    """Pickleball Highlight Tools"""
    pass


@cli.command("select-highlights")
@click.option("--input", "-i", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, type=click.Path())
@click.option("--max", "max_highlights", default=10, show_default=True)
def select_highlights_cmd(input, output, max_highlights):
    """Select top N non-overlapping highlights."""
    input_path = Path(input)
    output_path = Path(output)

    rallies = load_rallies(input_path)
    selected = select_top_highlights(rallies, max_highlights=max_highlights)

    output_path.write_text(json.dumps(export_rallies_to_dict(selected), indent=2))
    click.echo(f"Exported {len(selected)} highlights → {output_path}")


if __name__ == "__main__":
    cli()