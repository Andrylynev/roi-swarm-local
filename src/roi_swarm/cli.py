from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from openai import AsyncOpenAI
from rich.console import Console

from roi_swarm.gateway import ModelGateway
from roi_swarm.graph import build_graph
from roi_swarm.settings import load_effective_settings

app = typer.Typer(no_args_is_help=True)
console = Console()
DEFAULT_CONFIG = Path(__file__).parents[2] / "config" / "settings.yaml"


@app.command()
def run(
    request: Annotated[str, typer.Argument(help="Задача для адаптивного роя")],
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ] = DEFAULT_CONFIG,
) -> None:
    """Запустить адаптивный LangGraph-рой."""
    asyncio.run(_run(request, config))


@app.command("models")
def models(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ] = DEFAULT_CONFIG,
) -> None:
    """Проверить соединение и показать модели провайдера."""
    asyncio.run(_models(config))


async def _run(request: str, config: Path) -> None:
    settings = load_effective_settings(config)
    client = AsyncOpenAI(
        api_key=settings.provider.api_key,
        base_url=settings.provider.base_url,
        timeout=settings.provider.timeout_seconds,
        max_retries=settings.provider.max_retries,
    )
    try:
        graph = build_graph(
            settings,
            ModelGateway(client, api_mode=settings.provider.api_mode),
        )
        result = await graph.ainvoke(
            {"request": request},
            config={
                "configurable": {"thread_id": "cli"},
                "max_concurrency": settings.swarm.maximum_concurrency,
            },
        )
        console.print(result["final_answer"])
    finally:
        await client.close()


async def _models(config: Path) -> None:
    settings = load_effective_settings(config)
    client = AsyncOpenAI(
        api_key=settings.provider.api_key,
        base_url=settings.provider.base_url,
        timeout=settings.provider.timeout_seconds,
        max_retries=settings.provider.max_retries,
    )
    try:
        response = await client.models.list()
        for model in sorted(item.id for item in response.data):
            console.print(model)
    finally:
        await client.close()


if __name__ == "__main__":
    app()
