"""Native llama.cpp schema sampling for the independent local deployment."""

import json

from roi_swarm.gateway import ModelGateway
from roi_swarm.token_budget import check_context


class LocalModelGateway(ModelGateway):
    def __init__(self, client):
        super().__init__(client, api_mode="chat_completions")

    async def generate_structured(
        self,
        *,
        model,
        system,
        prompt,
        schema,
        max_output_tokens=4096,
        temperature=0.7,
        context_window_tokens=None,
        token_encoding="o200k_base",
    ):
        definition = schema.model_json_schema()
        messages = [
            {
                "role": "system",
                "content": system
                + "\nReturn only JSON matching the schema: "
                + json.dumps(definition, ensure_ascii=False),
            },
            {"role": "user", "content": prompt},
        ]
        if context_window_tokens is not None:
            check_context(
                messages,
                limit=context_window_tokens,
                output_tokens=max_output_tokens,
                encoding=token_encoding,
            )
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": definition,
                },
            },
            # 'none' exposes an empty <think> preamble to JSON grammar in this
            # native build; auto extracts it before schema-constrained sampling.
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_format": "auto",
            },
        )
        return schema.model_validate_json(response.choices[0].message.content or "")
