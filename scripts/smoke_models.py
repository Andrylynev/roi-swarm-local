import asyncio

from openai import AsyncOpenAI

from roi_swarm.gateway import ModelGateway
from roi_swarm.settings import load_settings


async def main() -> None:
    settings = load_settings("config/settings.yaml")
    client = AsyncOpenAI(
        api_key=settings.provider.api_key,
        base_url=settings.provider.base_url,
        timeout=settings.provider.timeout_seconds,
    )
    gateway = ModelGateway(client, api_mode=settings.provider.api_mode)
    luna = await gateway.generate_text(
        model=settings.model_for("router").model,
        system="Reply with exactly LUNA_OK",
        prompt="Health check",
        max_output_tokens=30,
    )
    terra = await gateway.generate_text(
        model=settings.model_for("planner").model,
        system="Reply with exactly TERRA_OK",
        prompt="Health check",
        max_output_tokens=30,
    )
    print(f"LUNA={luna}")
    print(f"TERRA={terra}")


asyncio.run(main())
