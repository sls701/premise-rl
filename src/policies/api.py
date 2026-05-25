"""
Frontier baseline policies.
Dispatches on config.provider in {openai, google}.
Fail-fast if the required API key is missing.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

SEARCH_TOOL_SCHEMA = {
    "name": "search_theorems",
    "description": (
        "Search the TheoremSearch corpus for mathematical statements. "
        "Returns the top-k results ranked by embedding similarity."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query",
            },
            "k": {
                "type": "integer",
                "description": "Number of results to return (default 10)",
                "default": 10,
            },
        },
        "required": ["query"],
    },
}


def _require_key(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            f"Set it in .env before running this baseline."
        )
    return val


# ---------------------------------------------------------------------------
# OpenAI agent
# ---------------------------------------------------------------------------

class OpenAIAgent:
    def __init__(self, model: str, temperature: float | None = 0.0, max_tokens: int = 1024):
        from openai import AsyncOpenAI
        _require_key("OPENAI_API_KEY")
        self._client = AsyncOpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def chat(self, messages: list[dict]) -> tuple[str | None, int | None]:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            tools=[{"type": "function", "function": SEARCH_TOOL_SCHEMA}],
            tool_choice="required",
            max_completion_tokens=self.max_tokens,
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        if msg.tool_calls:
            tc = msg.tool_calls[0]
            args = json.loads(tc.function.arguments)
            query = args.get("query", "")
            k = args.get("k", 10)
            return query, k

        return None, None


# ---------------------------------------------------------------------------
# Google (Gemini) agent — uses google-genai SDK
# ---------------------------------------------------------------------------

class GoogleAgent:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024):
        from google import genai
        from google.genai import types
        _require_key("GOOGLE_API_KEY")
        self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self._types = types
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def chat(self, messages: list[dict]) -> tuple[str | None, int | None]:
        import asyncio
        from google.genai import types

        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)

        # Convert to Gemini Content objects
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "user" if m["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        tool = types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="search_theorems",
                description=SEARCH_TOOL_SCHEMA["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(type="STRING", description="Natural-language search query"),
                        "k": types.Schema(type="INTEGER", description="Number of results"),
                    },
                    required=["query"],
                ),
            )
        ])

        config = types.GenerateContentConfig(
            system_instruction=system_msg,
            tools=[tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY"),
            ),
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            ),
        )

        for part in resp.candidates[0].content.parts:
            if part.function_call and part.function_call.name == "search_theorems":
                fc = part.function_call
                query = fc.args.get("query", "")
                k = int(fc.args.get("k", 10))
                return query, k

        return None, None


# ---------------------------------------------------------------------------
# AWS Bedrock agent (Claude via Converse API)
# ---------------------------------------------------------------------------

class BedrockAgent:
    def __init__(self, model: str, temperature: float = 1.0, max_tokens: int = 4096):
        import boto3
        region = os.environ.get("AWS_REGION", "us-west-2")
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def chat(self, messages: list[dict]) -> tuple[str | None, int | None]:
        import asyncio

        system_blocks = [{"text": m["content"]} for m in messages if m["role"] == "system"]
        converse_messages = [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in messages if m["role"] != "system"
        ]

        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "search_theorems",
                        "description": SEARCH_TOOL_SCHEMA["description"],
                        "inputSchema": {"json": SEARCH_TOOL_SCHEMA["parameters"]},
                    }
                }
            ],
            "toolChoice": {"any": {}},  # force a tool call every turn
        }

        kwargs = dict(
            modelId=self.model,
            messages=converse_messages,
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": self.temperature},
            toolConfig=tool_config,
        )
        if system_blocks:
            kwargs["system"] = system_blocks

        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: self._client.converse(**kwargs)
        )

        for block in resp["output"]["message"]["content"]:
            if block.get("toolUse") and block["toolUse"]["name"] == "search_theorems":
                tool_input = block["toolUse"]["input"]
                query = tool_input.get("query", "")
                k = int(tool_input.get("k", 10))
                return query, k

        return None, None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_agent(config: dict) -> "OpenAIAgent | GoogleAgent | BedrockAgent":
    provider = config.get("provider", "").lower()
    model = config["model"]
    temperature = config.get("temperature", 1.0)
    max_tokens = config.get("max_tokens", 4096)

    if provider == "openai":
        return OpenAIAgent(model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider == "google":
        return GoogleAgent(model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider == "bedrock":
        return BedrockAgent(model=model, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Must be 'openai', 'google', or 'bedrock'.")
