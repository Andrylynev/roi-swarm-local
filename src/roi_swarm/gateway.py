from __future__ import annotations

import json
import re
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from roi_swarm.token_budget import check_context

T = TypeVar("T", bound=BaseModel)


class ModelGateway:
    def __init__(self, client: AsyncOpenAI, api_mode: str = "responses") -> None:
        self.client = client
        self.api_mode = api_mode

    async def generate_text(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        max_output_tokens: int = 4096,
        temperature: float = 0.0,
        context_window_tokens: int | None = None,
        token_encoding: str = "o200k_base",
    ) -> str:
        if context_window_tokens is not None:
            check_context(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                limit=context_window_tokens,
                output_tokens=max_output_tokens,
                encoding=token_encoding,
            )
        if self.api_mode == "chat_completions":
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_output_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content
        else:
            response = await self.client.responses.create(
                model=model,
                instructions=system,
                input=prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            text = response.output_text
        if not text or not text.strip():
            raise ValueError(f"Model {model} returned empty output")
        return text.strip()

    async def generate_structured(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        schema: type[T],
        max_output_tokens: int = 4096,
        temperature: float = 0.0,
        context_window_tokens: int | None = None,
        token_encoding: str = "o200k_base",
    ) -> T:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        full_system = (
            f"{system}\nReturn only JSON matching this JSON Schema. "
            f"Do not add markdown. Schema: {schema_json}"
        )
        text = await self.generate_text(
            model=model,
            system=full_system,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            context_window_tokens=context_window_tokens,
            token_encoding=token_encoding,
        )
        try:
            return schema.model_validate_json(_extract_json(text))
        except ValueError as first_error:
            # Long prompts sometimes cause an incorrect JSON shape. Retry once.
            repair_prompt = (
                f"{prompt}\n\nYour previous reply did not match the required schema "
                f"({type(first_error).__name__}). Reply again with ONLY a JSON object "
                f"that has exactly these fields: {schema_json}"
            )
            text = await self.generate_text(
                model=model,
                system=full_system,
                prompt=repair_prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                context_window_tokens=context_window_tokens,
                token_encoding=token_encoding,
            )
            return schema.model_validate_json(_extract_json(text))


def _extract_json(text: str) -> str:
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    candidate = candidate.strip()
    json.loads(candidate)
    return candidate
